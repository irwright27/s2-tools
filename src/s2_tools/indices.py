from __future__ import annotations

import xarray as xr


def ndvi(
    nir: xr.DataArray,
    red: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).

    Parameters
    ----------
    nir : xarray.DataArray
        Near-infrared reflectance.
    red : xarray.DataArray
        Red reflectance.

    Returns
    -------
    xarray.DataArray
        NDVI with the same dimensions and coordinates as the inputs.
    """
    denominator = nir + red

    out = xr.where(
        denominator != 0,
        (nir - red) / denominator,
        float("nan"),
    )

    out.name = "NDVI"
    out.attrs = {
        "long_name": "Normalized Difference Vegetation Index",
        "valid_range": (-1.0, 1.0),
    }

    return out


def add_ndvi(
    ds: xr.Dataset,
    nir_band: str = "B08",
    red_band: str = "B04",
) -> xr.Dataset:
    """
    Calculate NDVI and add it to an xarray Dataset.
    """
    if nir_band not in ds:
        raise ValueError(f"Dataset does not contain '{nir_band}'.")

    if red_band not in ds:
        raise ValueError(f"Dataset does not contain '{red_band}'.")

    out = ds.copy()

    out["NDVI"] = ndvi(
        nir=out[nir_band],
        red=out[red_band],
    )

    return out