"""
Geographic helpers: bounding boxes and lon/lat <-> planar projections.

The binning and graph routines assume planar (x, y) coordinates in consistent units.
Geographic data is projected to an equal-area CRS (meters) before binning, and bin
centers are projected back to lon/lat for mapping.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    Transformer = None  # type: ignore
    HAS_PYPROJ = False

# (min_lon, min_lat, max_lon, max_lat)
BBox = tuple[float, float, float, float]

COLORADO_BBOX: BBox = (-109.06, 36.99, -102.04, 41.01)
DENVER_METRO_BBOX: BBox = (-105.30, 39.55, -104.60, 39.95)

NAMED_BBOXES: dict[str, BBox] = {
    "colorado": COLORADO_BBOX,
    "denver": DENVER_METRO_BBOX,
}

# CONUS Albers Equal Area (meters). Equal-area keeps hex bins comparable across the region.
DEFAULT_CRS = "EPSG:5070"


def parse_bbox(value: str) -> BBox:
    """
    Parses a bounding box from a region name ('colorado', 'denver') or 'min_lon,min_lat,max_lon,max_lat'.

    Args:
        value: Region name or comma-separated bounds.

    Returns:
        BBox: (min_lon, min_lat, max_lon, max_lat).

    Raises:
        ValueError: If the value is neither a known region nor four valid ordered numbers.
    """
    key = value.strip().lower()
    if key in NAMED_BBOXES:
        return NAMED_BBOXES[key]

    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Invalid bbox '{value}'. Use a region name ({', '.join(NAMED_BBOXES)}) "
            "or 'min_lon,min_lat,max_lon,max_lat'."
        )
    min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError(f"Invalid bbox '{value}': minimums must be smaller than maximums.")
    return (min_lon, min_lat, max_lon, max_lat)


def in_bbox(lon: np.ndarray, lat: np.ndarray, bbox: BBox) -> np.ndarray:
    """
    Returns a boolean mask of points that fall inside the bounding box (inclusive).

    Args:
        lon: 1D array of longitudes.
        lat: 1D array of latitudes.
        bbox: (min_lon, min_lat, max_lon, max_lat).

    Returns:
        np.ndarray: Boolean mask of shape (N,).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    return (lon >= min_lon) & (lon <= max_lon) & (lat >= min_lat) & (lat <= max_lat)


@lru_cache(maxsize=8)
def _transformer(src_crs: str, dst_crs: str) -> Transformer:
    if not HAS_PYPROJ or Transformer is None:
        raise ImportError("pyproj is required for projections. Install with `pip install pyproj`.")
    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def project_lonlat(lon: np.ndarray, lat: np.ndarray, crs: str = DEFAULT_CRS) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects WGS84 lon/lat (degrees) to planar coordinates in the target CRS.

    Args:
        lon: 1D array of longitudes.
        lat: 1D array of latitudes.
        crs: Target projected CRS (default: EPSG:5070, CONUS Albers, meters).

    Returns:
        tuple[np.ndarray, np.ndarray]: (x, y) arrays in CRS units.
    """
    x, y = _transformer("EPSG:4326", crs).transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    return np.asarray(x), np.asarray(y)


def unproject_xy(x: np.ndarray, y: np.ndarray, crs: str = DEFAULT_CRS) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts planar coordinates in the given CRS back to WGS84 lon/lat (degrees).

    Args:
        x: 1D array of x coordinates.
        y: 1D array of y coordinates.
        crs: Source projected CRS.

    Returns:
        tuple[np.ndarray, np.ndarray]: (lon, lat) arrays.
    """
    lon, lat = _transformer(crs, "EPSG:4326").transform(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return np.asarray(lon), np.asarray(lat)
