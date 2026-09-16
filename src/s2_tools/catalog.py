from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import planetary_computer
import pystac
from pystac_client import Client


PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

S2_COLLECTION = "sentinel-2-l2a"


def get_catalog() -> Client:
    """
    Open the Microsoft Planetary Computer STAC catalog.

    Returns
    -------
    pystac_client.Client
        Planetary Computer STAC client with automatic asset signing.
    """
    return Client.open(
        PC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def search_s2(
    aoi: Mapping[str, Any],
    start: str | date | datetime,
    end: str | date | datetime,
    max_cloud: float | None = 20,
) -> list[pystac.Item]:
    """
    Search Planetary Computer for Sentinel-2 Level-2A imagery.

    Parameters
    ----------
    aoi : mapping
        AOI geometry represented as a GeoJSON-like mapping.
    start : str, date, or datetime
        Start date of the search period.
    end : str, date, or datetime
        End date of the search period.
    max_cloud : float or None, default 20
        Maximum Sentinel-2 scene-level cloud cover percentage.
        Set to None to disable scene-level cloud filtering.

    Returns
    -------
    list[pystac.Item]
        Sentinel-2 STAC items intersecting the AOI and satisfying
        the requested date and cloud-cover filters.
    """
    if max_cloud is not None and not 0 <= max_cloud <= 100:
        raise ValueError("max_cloud must be between 0 and 100.")

    catalog = get_catalog()

    query = None

    if max_cloud is not None:
        query = {
            "eo:cloud_cover": {
                "lte": max_cloud,
            }
        }

    search = catalog.search(
        collections=[S2_COLLECTION],
        intersects=aoi,
        datetime=f"{start}/{end}",
        query=query,
    )

    items = list(search.items())
    items.sort(key=lambda item: item.datetime)

    return items