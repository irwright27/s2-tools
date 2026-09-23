from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_GAMMA = 1.0


def isolate_olive_ndvi(
    stl_result: pd.DataFrame,
    gamma: float = DEFAULT_GAMMA,
) -> pd.DataFrame:
    """
    Estimate olive-associated and cover-associated NDVI from an STL result.

    This is a deliberately simple signal-partitioning approach intended to
    produce a Sentinel-2 olive-associated NDVI proxy that can be compared with
    olive-only NDVI derived from UAV canopy segmentation.

    Procedure
    ---------
    1. Adjust the STL seasonal component by retaining positive remainder:

           seasonal_adjusted = max(seasonal, seasonal + remainder)

    2. Normalize the adjusted seasonal component to a 0-1 phenology function:

           amplitude = peak - trough
           S(t) = (seasonal_adjusted - trough) / amplitude
           s = mean(S)

    3. Estimate a persistent/background VI baseline:

           ndvi_base(t) = trend(t) - s * amplitude

    4. Calculate the departure of observed mixed-pixel NDVI from that baseline:

           anomaly(t) = observed(t) - ndvi_base(t)

    5. Attribute the anomaly between cover and olive using S(t):

           w_cover(t) = S(t) ** gamma
           w_olive(t) = 1 - w_cover(t)

           ndvi_cover(t) = w_cover(t) * anomaly(t)
           ndvi_olive(t) = ndvi_base(t) + w_olive(t) * anomaly(t)

    By construction:

           observed = ndvi_olive + ndvi_cover

    Notes
    -----
    ``ndvi_olive`` is an STL-derived olive-associated NDVI proxy. It is not
    assumed to be physically identical to NDVI calculated from spatially
    isolated olive pixels. Comparison with UAV olive-only NDVI is therefore
    an empirical validation of the proxy. NDVI is nonlinear and not physically
    additive between subpixel vegetation classes: closure here is a model
    bookkeeping identity, not a radiative-transfer statement. Final NDVI
    outputs are not clipped. A zero adjusted-seasonal amplitude sets S=0,
    assigning the observed signal entirely to the olive-associated proxy.

    Parameters
    ----------
    stl_result : pandas.DataFrame
        Output from s2_tools.stl.decompose_stl(). Must contain observed,
        trend, seasonal, and remainder.
    gamma : float, default 1.0
        Controls how strongly the phenology function assigns the anomaly to
        cover vegetation. gamma=1 uses S directly.

    Returns
    -------
    pandas.DataFrame
        Original STL columns plus seasonal_adjusted, S, ndvi_base, anomaly,
        w_cover, w_olive, ndvi_cover, and ndvi_olive.
    """
    if not isinstance(stl_result, pd.DataFrame):
        raise TypeError("stl_result must be a pandas DataFrame.")

    if stl_result.empty:
        raise ValueError("stl_result is empty.")

    required = {"observed", "trend", "seasonal", "remainder"}
    missing = required.difference(stl_result.columns)
    if missing:
        raise ValueError(
            f"stl_result is missing required columns: {sorted(missing)}"
        )

    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be a finite number > 0.")

    core = stl_result[list(required)].to_numpy(dtype=float)
    if not np.isfinite(core).all():
        raise ValueError("stl_result contains NaN or infinite values.")

    out = stl_result.copy()

    seasonal_adjusted = np.maximum(
        out["seasonal"].to_numpy(dtype=float),
        (
            out["seasonal"] + out["remainder"]
        ).to_numpy(dtype=float),
    )
    seasonal_adjusted = pd.Series(
        seasonal_adjusted,
        index=out.index,
        name="seasonal_adjusted",
    )

    trough = float(seasonal_adjusted.min())
    peak = float(seasonal_adjusted.max())
    amplitude = peak - trough

    if not np.isfinite(amplitude) or amplitude <= 0:
        S = pd.Series(0.0, index=out.index, name="S")
        s = 0.0
    else:
        S = ((seasonal_adjusted - trough) / amplitude).clip(0.0, 1.0)
        S.name = "S"
        s = float(S.mean())

    ndvi_base = out["trend"] - s * amplitude
    anomaly = out["observed"] - ndvi_base

    w_cover = (S ** gamma).clip(0.0, 1.0)
    w_olive = 1.0 - w_cover

    ndvi_cover = w_cover * anomaly
    ndvi_olive = ndvi_base + w_olive * anomaly

    out["seasonal_adjusted"] = seasonal_adjusted
    out["S"] = S
    out["ndvi_base"] = ndvi_base
    out["anomaly"] = anomaly
    out["w_cover"] = w_cover
    out["w_olive"] = w_olive
    out["ndvi_cover"] = ndvi_cover
    out["ndvi_olive"] = ndvi_olive

    out.attrs.update(stl_result.attrs)
    out.attrs["partition"] = {
        "gamma": float(gamma),
        "trough": trough,
        "peak": peak,
        "amplitude": float(amplitude),
        "s": float(s),
    }

    return out
