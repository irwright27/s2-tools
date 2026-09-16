from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import odc.stac
import pystac
import xarray as xr


DEFAULT_BANDS = ("B04", "B08", "SCL")


def load_s2(
    items: Sequence[pystac.Item],
    aoi: Mapping[str, Any],
    bands: Sequence[str] = DEFAULT_BANDS,
    resolution: float = 10,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """
    Load Sentinel-2 STAC items into an xarray Dataset.

    Parameters
    ----------
    items : sequence of pystac.Item
        Sentinel-2 STAC items, typically returned by search_s2().
    aoi : mapping
        AOI geometry represented as a GeoJSON-like mapping in EPSG:4326.
    bands : sequence of str, default ("B04", "B08", "SCL")
        Sentinel-2 assets to load.
    resolution : float, default 10
        Output spatial resolution in meters.
    chunks : dict or None
        Dask chunk sizes. If None, a default spatial chunking scheme
        is used.

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions time, y, and x.
    """
    if not items:
        raise ValueError("No STAC items were provided.")

    if resolution <= 0:
        raise ValueError("resolution must be greater than zero.")

    if chunks is None:
        chunks = {
            "time": 1,
            "x": 1024,
            "y": 1024,
        }

    ds = odc.stac.load(
        items,
        bands=list(bands),
        geopolygon=aoi,
        resolution=resolution,
        chunks=chunks,
    )

    return ds