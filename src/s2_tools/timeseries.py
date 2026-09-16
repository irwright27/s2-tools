from __future__ import annotations

import pandas as pd
import xarray as xr


def normalize_dates(
    ds: xr.Dataset,
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Normalize acquisition timestamps to calendar dates.

    For example:

        2024-06-07T18:49:19 -> 2024-06-07T00:00:00

    No observations are added or removed.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a time coordinate.

    time_dim : str, default "time"
        Name of the time dimension.

    Returns
    -------
    xarray.Dataset
        Dataset with timestamps normalized to midnight.
    """
    if time_dim not in ds.coords:
        raise ValueError(
            f"Dataset does not contain a '{time_dim}' coordinate."
        )

    out = ds.copy()

    dates = pd.DatetimeIndex(out[time_dim].values).normalize()

    # Normalizing timestamps could theoretically create duplicate dates.
    if dates.duplicated().any():
        duplicates = dates[dates.duplicated()].unique()

        raise ValueError(
            "Normalizing timestamps would create duplicate dates: "
            f"{list(duplicates)}"
        )

    out = out.assign_coords({time_dim: dates})

    return out


def regularize_time(
    ds: xr.Dataset,
    freq: str = "1D",
    time_dim: str = "time",
) -> xr.Dataset:
    """
    Reindex a dataset onto a regular temporal grid.

    Missing dates are inserted but are not interpolated. Variables on
    missing dates therefore contain NaN.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a time coordinate.

    freq : str, default "1D"
        Pandas-compatible temporal frequency.

    time_dim : str, default "time"
        Name of the time dimension.

    Returns
    -------
    xarray.Dataset
        Dataset reindexed onto a regular temporal grid.
    """
    if time_dim not in ds.coords:
        raise ValueError(
            f"Dataset does not contain a '{time_dim}' coordinate."
        )

    if ds.sizes.get(time_dim, 0) == 0:
        raise ValueError("Dataset contains no time observations.")

    start = pd.Timestamp(ds[time_dim].min().item())
    end = pd.Timestamp(ds[time_dim].max().item())

    new_time = pd.date_range(
        start=start,
        end=end,
        freq=freq,
    )

    return ds.reindex({time_dim: new_time})