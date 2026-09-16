from __future__ import annotations

from collections.abc import Sequence

import xarray as xr


# Sentinel-2 Scene Classification Layer (SCL) classes that are
# considered invalid for surface-reflectance analysis.
DEFAULT_INVALID_SCL = (
    0,   # No data
    1,   # Saturated or defective
    3,   # Cloud shadow
    8,   # Cloud, medium probability
    9,   # Cloud, high probability
    10,  # Thin cirrus
)


def scl_valid_mask(
    scl: xr.DataArray,
    invalid_classes: Sequence[int] = DEFAULT_INVALID_SCL,
) -> xr.DataArray:
    """
    Create a boolean validity mask from Sentinel-2 SCL data.

    Parameters
    ----------
    scl : xarray.DataArray
        Sentinel-2 Scene Classification Layer.
    invalid_classes : sequence of int
        SCL classes to consider invalid.

    Returns
    -------
    xarray.DataArray
        Boolean array where True indicates a valid pixel.
    """
    return ~scl.isin(invalid_classes)


def mask_s2(
    ds: xr.Dataset,
    bands: Sequence[str] | None = None,
    invalid_classes: Sequence[int] = DEFAULT_INVALID_SCL,
) -> xr.Dataset:
    """
    Mask invalid Sentinel-2 pixels using the Scene Classification Layer.

    Invalid pixels are replaced with NaN in the requested spectral bands.
    The SCL layer itself is retained unchanged.

    Parameters
    ----------
    ds : xarray.Dataset
        Sentinel-2 dataset containing an SCL variable.

    bands : sequence of str or None
        Variables to mask. If None, all variables except SCL are masked.

    invalid_classes : sequence of int
        SCL classes to consider invalid.

    Returns
    -------
    xarray.Dataset
        Dataset with invalid spectral pixels replaced by NaN.
    """
    if "SCL" not in ds:
        raise ValueError("Dataset must contain an 'SCL' variable.")

    valid = scl_valid_mask(
        ds["SCL"],
        invalid_classes=invalid_classes,
    )

    if bands is None:
        bands = [
            name
            for name in ds.data_vars
            if name != "SCL"
        ]

    out = ds.copy()

    for band in bands:
        if band not in ds:
            raise ValueError(f"Dataset does not contain band '{band}'.")

        out[band] = ds[band].where(valid)

    return out