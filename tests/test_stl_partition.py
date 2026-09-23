"""Regression checks for daily preparation, STL constraints, and proxy bookkeeping."""
import unittest

import numpy as np
import pandas as pd

from s2_tools.partition import isolate_olive_ndvi
from s2_tools.stl import decompose_stl, prepare_daily_series, stl_parameters


class STLPartitionTests(unittest.TestCase):
    def test_same_day_mean_and_linear_interpolation(self):
        raw = pd.Series([0.6, 0.2, 0.8], index=pd.to_datetime([
            "2024-01-01 18:00Z", "2024-01-01 10:00Z", "2024-01-03 12:00Z"
        ]))
        daily = prepare_daily_series(raw)
        np.testing.assert_allclose(daily, [0.4, 0.6, 0.8])
        self.assertEqual(str(daily.index.tz), "UTC")
        self.assertTrue(np.isnan(prepare_daily_series(raw, interpolate=False).iloc[1]))

    def test_invalid_input_and_parameters(self):
        for kwargs in [{"ns": 8}, {"ns": 7.0}, {"period": True}, {"inner_iter": 1.5}, {"outer_iter": -1}]:
            with self.assertRaises(ValueError):
                stl_parameters(**kwargs)
        irregular = pd.Series([0.3, 0.4], index=pd.to_datetime(["2024-01-01", "2024-01-03"]))
        with self.assertRaises(ValueError):
            decompose_stl(irregular)
        for values in [[np.nan, np.nan], [0.3, np.inf]]:
            with self.assertRaises(ValueError):
                prepare_daily_series(pd.Series(values, index=pd.date_range("2024-01-01", periods=2)))
        with self.assertRaises(ValueError):
            isolate_olive_ndvi(pd.DataFrame(columns=["observed", "trend", "seasonal", "remainder"]))

    def test_candidates_and_closure(self):
        dates = pd.date_range("2021-01-01", "2025-12-31")
        values = 0.4 + 0.1 * np.cos(2 * np.pi * np.arange(len(dates)) / 365)
        values += np.where(dates < "2022-01-01", 0.2, 0.0)
        series = pd.Series(values, index=dates)
        for ns in [7, 15, 31, 61, 91]:
            result = decompose_stl(series, ns=ns)
            params = result.attrs["stl_parameters"]
            self.assertEqual(params["low_pass"], 367)
            self.assertEqual(params["trend"] % 2, 1)
            minimum = 1.5 * 365 / (1 - 1.5 / ns)
            self.assertGreaterEqual(params["trend"], minimum)
            self.assertLess(params["trend"] - 2, minimum)
            proxy = isolate_olive_ndvi(result)
            np.testing.assert_allclose(result[["trend", "seasonal", "remainder"]].sum(axis=1), series, atol=1e-12)
            np.testing.assert_allclose(proxy["ndvi_olive"] + proxy["ndvi_cover"], series, atol=1e-12)
            self.assertTrue(proxy["S"].between(0, 1).all())

    def test_partition_formula_and_zero_amplitude_no_clipping(self):
        frame = pd.DataFrame({"observed": [0.3, 1.3, 0.2], "trend": [0.4]*3,
                              "seasonal": [-0.1, 0.2, 0.0], "remainder": [0.0, 0.7, -0.2]})
        proxy = isolate_olive_ndvi(frame, gamma=2)
        adjusted = np.array([-0.1, 0.9, 0.0])
        S = (adjusted - adjusted.min()) / np.ptp(adjusted)
        base = frame["trend"] - S.mean() * np.ptp(adjusted)
        np.testing.assert_allclose(proxy["seasonal_adjusted"], adjusted)
        np.testing.assert_allclose(proxy["ndvi_cover"], S**2 * (frame["observed"] - base))
        constant = pd.DataFrame({"observed": [1.2]*3, "trend": [1.2]*3,
                                 "seasonal": [0.0]*3, "remainder": [0.0]*3})
        zero = isolate_olive_ndvi(constant)
        np.testing.assert_allclose(zero["ndvi_olive"], 1.2)
        np.testing.assert_allclose(zero["ndvi_cover"], 0.0)
        self.assertEqual(zero.attrs["partition"]["amplitude"], 0.0)


if __name__ == "__main__":
    unittest.main()
