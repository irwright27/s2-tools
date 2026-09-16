from __future__ import annotations

import math

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


DEFAULT_PERIOD = 365
DEFAULT_INNER_ITER = 2
DEFAULT_OUTER_ITER = 5


def next_odd(value: float) -> int:
    """
    Return the smallest odd integer greater than or equal to value.
    """
    n = math.ceil(value)

    if n % 2 == 0:
        n += 1

    return n


def calculate_nt(
    ns: int,
    period: int = DEFAULT_PERIOD,
) -> int:
    """
    Calculate the STL trend-smoother length following the
    recommendation of Cleveland et al. (1990).

    Parameters
    ----------
    ns : int
        Seasonal LOESS smoothing parameter.

    period : int, default 365
        Number of observations in one seasonal cycle.

    Returns
    -------
    int
        Smallest odd integer satisfying Cleveland's recommended
        lower bound for the trend smoother.
    """
    if period < 2:
        raise ValueError("period must be at least 2.")

    if ns < 3:
        raise ValueError("ns must be at least 3.")

    if ns % 2 == 0:
        raise ValueError("ns must be an odd integer.")

    denominator = 1 - (1.5 / ns)

    nt_min = (1.5 * period) / denominator

    return next_odd(nt_min)


def calculate_nl(
    period: int = DEFAULT_PERIOD,
) -> int:
    """
    Calculate the STL low-pass smoother length.

    nl is the smallest odd integer strictly greater than the
    seasonal period, consistent with the STL implementation
    used by statsmodels.
    """
    if period < 2:
        raise ValueError("period must be at least 2.")

    nl = period + 1

    if nl % 2 == 0:
        nl += 1

    return nl


def stl_parameters(
    ns: int,
    period: int = DEFAULT_PERIOD,
    inner_iter: int = DEFAULT_INNER_ITER,
    outer_iter: int = DEFAULT_OUTER_ITER,
) -> dict[str, int]:
    """
    Construct the STL parameter set used by s2-tools.

    The seasonal smoother ns is selected diagnostically.
    The trend and low-pass smoother lengths are calculated
    following Cleveland et al. (1990).
    """
    if inner_iter < 1:
        raise ValueError("inner_iter must be at least 1.")

    if outer_iter < 0:
        raise ValueError("outer_iter cannot be negative.")

    return {
        "period": period,
        "seasonal": ns,
        "trend": calculate_nt(
            ns=ns,
            period=period,
        ),
        "low_pass": calculate_nl(
            period=period,
        ),
        "inner_iter": inner_iter,
        "outer_iter": outer_iter,
    }


def decompose_stl(
    series: pd.Series,
    ns: int,
    period: int = DEFAULT_PERIOD,
    inner_iter: int = DEFAULT_INNER_ITER,
    outer_iter: int = DEFAULT_OUTER_ITER,
) -> pd.DataFrame:
    """
    Decompose a regularly spaced time series using STL.

    Parameters
    ----------
    series : pandas.Series
        Complete, regularly spaced time series with a DatetimeIndex.
        Missing values are not allowed.

    ns : int
        Seasonal LOESS smoothing parameter. This parameter should
        be selected diagnostically.

    period : int, default 365
        Number of observations in one seasonal cycle.

    inner_iter : int, default 2
        Number of STL inner-loop iterations.

    outer_iter : int, default 5
        Number of STL robustness iterations.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the original series and its STL
        trend, seasonal, and remainder components.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(
            "series must have a pandas DatetimeIndex."
        )

    if series.empty:
        raise ValueError("series is empty.")

    values = series.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "series contains NaN or infinite values. "
            "Gap filling must be performed before STL decomposition."
        )

    params = stl_parameters(
        ns=ns,
        period=period,
        inner_iter=inner_iter,
        outer_iter=outer_iter,
    )

    model = STL(
        series,
        period=params["period"],
        seasonal=params["seasonal"],
        trend=params["trend"],
        low_pass=params["low_pass"],
        robust=outer_iter > 0,
    )

    result = model.fit(
        inner_iter=inner_iter,
        outer_iter=outer_iter,
    )

    return pd.DataFrame(
        {
            "observed": series,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "remainder": result.resid,
        },
        index=series.index,
    )