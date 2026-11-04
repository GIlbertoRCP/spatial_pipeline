"""
Geotagged photo ingestion: YFCC100M metadata readers, cleaning, and tag extraction.

YFCC100M metadata ships as tab-separated rows (23 fields in the original Yahoo Webscope
release, 25 in the AWS Multimedia Commons release, which prepends a line number and adds a
photo hash) or as a SQLite database with the same fields. Columns are mapped by position,
so the readers work regardless of how a given copy names them.
"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd

from .geo import BBox, in_bbox

YFCC_FIELDS_23 = [
    "photo_id", "user_id", "user_nickname", "date_taken", "date_uploaded", "capture_device",
    "title", "description", "user_tags", "machine_tags", "longitude", "latitude", "accuracy",
    "page_url", "download_url", "license_name", "license_url", "server_id", "farm_id",
    "secret", "secret_original", "extension_original", "marker",
]
YFCC_FIELDS_25 = ["line_number", "photo_id", "photo_hash"] + YFCC_FIELDS_23[1:]

_FIELDS_BY_WIDTH = {23: YFCC_FIELDS_23, 25: YFCC_FIELDS_25}
_NEEDED = [
    "photo_id", "user_id", "user_nickname", "date_taken", "title", "user_tags", "longitude", "latitude",
    "accuracy", "page_url", "download_url", "license_name", "marker",
]

PHOTO_COLUMNS = ["photo_id", "user_id", "lon", "lat", "accuracy", "taken_at", "tags"]

# Flickr static image URLs: .../<server>/<id>_<secret>.jpg -> 150px square thumbnail.
_FLICKR_IMAGE = r"^https?://[^/]+/(?P<server>\d+)/(?P<id>\d+)_(?P<secret>[0-9a-f]+)\.jpg$"

# Tags that describe the camera, app or upload rather than the scene or place.
DEFAULT_STOP_TAGS = frozenset({
    "iphone", "iphoneography", "instagram", "instagramapp", "square", "squareformat",
    "uploaded:by=instagram", "foursquare", "android", "samsung", "nikon", "canon", "sony",
    "fuji", "olympus", "panasonic", "pentax", "leica", "dslr", "slr", "camera", "photo",
    "photos", "photography", "photographer", "picture", "pictures", "image", "flickr",
    "geotagged", "geotag", "geo", "gps", "hdr", "lightroom", "photoshop", "edited",
    "nofilter", "vsco", "vscocam", "jpg", "raw", "digital", "flickrandroidapp",
})


def _fields_for_width(width: int) -> list[str]:
    if width not in _FIELDS_BY_WIDTH:
        raise ValueError(
            f"Unrecognized YFCC100M layout with {width} columns (expected 23 or 25)."
        )
    return _FIELDS_BY_WIDTH[width]


def decode_tags(raw: pd.Series) -> pd.Series:
    """
    Decodes YFCC comma-separated, URL-encoded tag strings ('rocky+mountains,colorado').

    Args:
        raw: Series of raw tag strings (may contain empty strings / NaN).

    Returns:
        pd.Series: Lowercased, comma-separated decoded tags ('' when a photo has none).
    """
    raw = raw.fillna("").astype(str)
    uniques = pd.unique(raw)
    decoded = {
        value: ",".join(t for t in (unquote_plus(part).strip().lower() for part in value.split(",")) if t)
        for value in uniques
    }
    return raw.map(decoded)


def clean_yfcc_frame(
    df: pd.DataFrame,
    bbox: BBox | None = None,
    min_accuracy: int = 12,
    include_videos: bool = False,
) -> pd.DataFrame:
    """
    Filters and normalizes raw YFCC metadata rows into the standard photo table.

    Args:
        df: Raw rows with (at least) the YFCC field names in `_NEEDED`, as strings.
        bbox: Optional (min_lon, min_lat, max_lon, max_lat) filter.
        min_accuracy: Minimum Flickr geotag accuracy (1 = world ... 16 = street level). 0 disables.
        include_videos: Keep videos as well as photos.

    Returns:
        pd.DataFrame: Columns ['photo_id', 'user_id', 'lon', 'lat', 'accuracy', 'taken_at', 'tags',
        'owner_name', 'title', 'license', 'thumbnail_url', 'page_url'].
    """
    if not include_videos:
        df = df[df["marker"].astype(str).str.strip() == "0"]

    lon = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(dtype=np.float64)
    lat = pd.to_numeric(df["latitude"], errors="coerce").to_numpy(dtype=np.float64)
    accuracy = pd.to_numeric(df["accuracy"], errors="coerce").fillna(0).to_numpy(dtype=np.int16)

    keep = ~(np.isnan(lon) | np.isnan(lat))
    if bbox is not None:
        keep &= in_bbox(np.nan_to_num(lon), np.nan_to_num(lat), bbox)
    if min_accuracy > 0:
        keep &= accuracy >= min_accuracy

    df = df[keep]
    out = pd.DataFrame({
        "photo_id": df["photo_id"].astype(str).to_numpy(),
        "user_id": df["user_id"].astype(str).to_numpy(),
        "lon": lon[keep],
        "lat": lat[keep],
        "accuracy": accuracy[keep],
        "taken_at": pd.to_datetime(df["date_taken"], errors="coerce", format="ISO8601").to_numpy(),
        "tags": decode_tags(df["user_tags"]).to_numpy(),
        "owner_name": _decode(df["user_nickname"]),
        "title": _decode(df["title"]),
        "license": _decode(df["license_name"]),
        "thumbnail_url": _flickr_thumbnail(df["download_url"]),
        "page_url": df["page_url"].fillna("").astype(str).to_numpy(),
    })
    return out


def _decode(raw: pd.Series) -> np.ndarray:
    raw = raw.fillna("").astype(str)
    uniques = pd.unique(raw)
    decoded = {value: unquote_plus(value) for value in uniques}
    return np.asarray(raw.map(decoded).to_numpy())


def _flickr_thumbnail(download_url: pd.Series) -> np.ndarray:
    parts = download_url.fillna("").astype(str).str.extract(_FLICKR_IMAGE)
    thumb = "https://live.staticflickr.com/" + parts["server"] + "/" + parts["id"] + "_" + parts["secret"] + "_q.jpg"
    return np.asarray(thumb.fillna("").to_numpy())


def _iter_tsv_chunks(path: str, chunksize: int) -> Iterator[pd.DataFrame]:
    read_opts = {
        "sep": "\t",
        "header": None,
        "quoting": csv.QUOTE_NONE,
        "dtype": str,
        "keep_default_na": False,
        "compression": "infer",
    }
    first = pd.read_csv(path, nrows=1, **read_opts)  # type: ignore[call-overload]
    fields = _fields_for_width(first.shape[1])
    usecols = [fields.index(name) for name in _NEEDED]

    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, **read_opts)  # type: ignore[call-overload]
    for chunk in reader:
        chunk.columns = [fields[i] for i in sorted(usecols)]
        yield chunk


def _find_sqlite_table(conn: sqlite3.Connection) -> tuple[str, list[str]]:
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    for table in tables:
        columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
        if len(columns) in _FIELDS_BY_WIDTH:
            return table, columns
    raise ValueError(f"No table with a 23- or 25-column YFCC100M layout found (tables: {tables}).")


def _iter_sqlite_chunks(path: str, chunksize: int, bbox: BBox | None) -> Iterator[pd.DataFrame]:
    conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        table, columns = _find_sqlite_table(conn)
        fields = _fields_for_width(len(columns))
        actual = {name: columns[fields.index(name)] for name in _NEEDED}
        select = ", ".join(f'"{actual[name]}" AS {name}' for name in _NEEDED)
        query = f'SELECT {select} FROM "{table}"'
        params: tuple[float, ...] = ()
        if bbox is not None:
            # Pre-filter in SQLite so only rows near the region reach pandas.
            lon_col, lat_col = actual["longitude"], actual["latitude"]
            query += (
                f' WHERE "{lon_col}" != \'\' AND "{lat_col}" != \'\''
                f' AND CAST("{lon_col}" AS REAL) BETWEEN ? AND ?'
                f' AND CAST("{lat_col}" AS REAL) BETWEEN ? AND ?'
            )
            params = (bbox[0], bbox[2], bbox[1], bbox[3])
        for chunk in pd.read_sql_query(query, conn, params=params, chunksize=chunksize):
            yield chunk.astype(str)
    finally:
        conn.close()


def _is_sqlite(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(16) == b"SQLite format 3\x00"


def read_yfcc_metadata(
    path: str,
    bbox: BBox | None = None,
    min_accuracy: int = 12,
    include_videos: bool = False,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Streams YFCC100M metadata (TSV, .bz2/.gz TSV, or SQLite) and returns cleaned, deduplicated photos.

    Only one chunk of raw rows is held in memory at a time, so the full 100M-row dataset can be
    filtered down to a region on a laptop.

    Args:
        path: Path to the metadata file.
        bbox: Optional (min_lon, min_lat, max_lon, max_lat) filter.
        min_accuracy: Minimum Flickr geotag accuracy (1 = world ... 16 = street level). 0 disables.
        include_videos: Keep videos as well as photos.
        chunksize: Raw rows read per chunk.

    Returns:
        pd.DataFrame: Columns ['photo_id', 'user_id', 'lon', 'lat', 'accuracy', 'taken_at', 'tags',
        'owner_name', 'title', 'license', 'thumbnail_url', 'page_url'].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not match a known YFCC100M layout.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")

    chunks = _iter_sqlite_chunks(path, chunksize, bbox) if _is_sqlite(path) else _iter_tsv_chunks(path, chunksize)
    cleaned = [clean_yfcc_frame(chunk, bbox, min_accuracy, include_videos) for chunk in chunks]
    cleaned = [c for c in cleaned if len(c)]
    if not cleaned:
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in _empty_dtypes().items()})

    photos = pd.concat(cleaned, ignore_index=True)
    return photos.drop_duplicates("photo_id").reset_index(drop=True)


def _empty_dtypes() -> dict[str, str]:
    return {
        "photo_id": "object", "user_id": "object", "lon": "float64", "lat": "float64",
        "accuracy": "int16", "taken_at": "datetime64[ns]", "tags": "object", "owner_name": "object",
        "title": "object", "license": "object", "thumbnail_url": "object", "page_url": "object",
    }


def save_photos(photos: pd.DataFrame, path: str) -> None:
    """
    Saves a photo table to Parquet (recommended) or CSV, chosen by file extension.

    Args:
        photos: Photo table.
        path: Output path ending in .parquet or .csv.
    """
    if path.endswith(".csv"):
        photos.to_csv(path, index=False)
    else:
        photos.to_parquet(path, index=False)


def load_photos(path: str) -> pd.DataFrame:
    """
    Loads a photo table written by `save_photos`.

    Args:
        path: Path to a .parquet or .csv photo table.

    Returns:
        pd.DataFrame: Photo table with the standard columns.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".csv"):
        photos = pd.read_csv(path, dtype={"photo_id": str, "user_id": str, "tags": str}, keep_default_na=False)
        photos["taken_at"] = pd.to_datetime(photos["taken_at"], errors="coerce")
    else:
        photos = pd.read_parquet(path)
    missing = set(PHOTO_COLUMNS) - set(photos.columns)
    if missing:
        raise ValueError(f"Photo table {path} is missing columns: {sorted(missing)}")
    return photos


def explode_tags(
    photos: pd.DataFrame,
    min_length: int = 3,
    max_length: int = 40,
    stop_tags: frozenset[str] | set[str] = DEFAULT_STOP_TAGS,
) -> pd.DataFrame:
    """
    Expands the comma-separated `tags` column into one row per (photo, tag).

    Drops tags that are too short/long, purely numeric (years, dates), machine-tag-like
    ('key:value' / 'key=value'), or in `stop_tags`. Duplicate tags on the same photo are removed.

    Args:
        photos: Photo table with a `tags` column.
        min_length: Minimum tag length.
        max_length: Maximum tag length.
        stop_tags: Tags to ignore.

    Returns:
        pd.DataFrame: One row per photo-tag with the photo's index as `photo_idx` plus a `label` column.
    """
    tags = photos["tags"].fillna("").astype(str).str.split(",").explode()
    tags = tags[tags.str.len().between(min_length, max_length)]
    tags = tags[~tags.str.fullmatch(r"\d+") & ~tags.str.contains(r"[:=]", regex=True)]
    if stop_tags:
        tags = tags[~tags.isin(stop_tags)]

    out = pd.DataFrame({"photo_idx": tags.index.to_numpy(), "label": tags.to_numpy()})
    return out.drop_duplicates().reset_index(drop=True)
