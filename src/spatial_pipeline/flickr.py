"""
Flickr API photo fetcher: geotagged photos for a region, in the standard photo table format.

`flickr.photos.search` has two quirks that shape this module:

- A geo query without another limiting filter only returns photos from the last 12 hours,
  so every query carries an upload-date window.
- A single query returns at most ~4,000 distinct results, however many pages exist. Windows
  whose total exceeds that cap are split, first in time and then in space (bbox quadrants),
  until every window fits.

Each finished window is cached as a Parquet part, so an interrupted fetch resumes where it
stopped. Requests are throttled to stay under Flickr's limit of 3,600 calls per key per hour.

Flickr now issues API keys only to Pro (paid) accounts; `inat` is the free alternative.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ._http import ThrottledJSONClient
from .geo import BBox, in_bbox

logger = logging.getLogger(__name__)

API_URL = "https://api.flickr.com/services/rest/"
RESULT_CAP = 4000
MAX_PER_PAGE = 500
EXTRAS = "geo,tags,date_taken,date_upload,owner_name,license,url_q"

# Flickr license ids that allow reuse (Creative Commons, public domain, no known restrictions).
# See flickr.photos.licenses.getInfo for the current list.
OPEN_LICENSES = "1,2,3,4,5,6,7,8,9,10"

FLICKR_PHOTO_COLUMNS = [
    "photo_id", "user_id", "lon", "lat", "accuracy", "taken_at", "tags",
    "uploaded_at", "owner_name", "title", "license", "thumbnail_url", "page_url",
]


class FlickrError(RuntimeError):
    """Raised when the Flickr API returns an error response."""


class SearchClient(Protocol):
    def search(self, params: dict[str, str]) -> dict[str, Any]: ...


class FlickrClient(ThrottledJSONClient):
    """
    Minimal throttled `flickr.photos.search` client with retries.

    Note: Flickr now issues API keys only to Pro (paid) accounts.

    Args:
        api_key: Flickr API key (create one at https://www.flickr.com/services/apps/create/).
        min_interval: Minimum seconds between requests (1.0 = 3,600 requests/hour).
        max_retries: Retries for network errors, HTTP 429 and 5xx responses.
        timeout: Per-request timeout in seconds.
        sleep: Sleep function (injectable for tests).
        opener: Callable taking a urllib Request and returning a response (injectable for tests).
    """

    def __init__(
        self,
        api_key: str,
        min_interval: float = 1.0,
        max_retries: int = 4,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("A Flickr API key is required (set FLICKR_API_KEY).")
        super().__init__(min_interval=min_interval, max_retries=max_retries, timeout=timeout, sleep=sleep, opener=opener)
        self.api_key = api_key

    def search(self, params: dict[str, str]) -> dict[str, Any]:
        """
        Calls `flickr.photos.search` and returns the parsed `photos` object.

        Args:
            params: Search parameters (method, key and format are added automatically).

        Returns:
            dict[str, Any]: The response's `photos` object (page, pages, total, photo list).

        Raises:
            FlickrError: If Flickr reports an error (bad key, invalid parameters, ...).
        """
        payload = self.get_json(API_URL, {
            **params,
            "method": "flickr.photos.search",
            "api_key": self.api_key,
            "format": "json",
            "nojsoncallback": "1",
        })
        if payload.get("stat") != "ok":
            raise FlickrError(f"Flickr API error {payload.get('code')}: {payload.get('message')}")
        photos: dict[str, Any] = payload["photos"]
        return photos


def _split_bbox(bbox: BBox) -> list[BBox]:
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lon = (min_lon + max_lon) / 2.0
    mid_lat = (min_lat + max_lat) / 2.0
    return [
        (min_lon, min_lat, mid_lon, mid_lat),
        (mid_lon, min_lat, max_lon, mid_lat),
        (min_lon, mid_lat, mid_lon, max_lat),
        (mid_lon, mid_lat, max_lon, max_lat),
    ]


def _window_key(bbox: BBox, start: int, end: int) -> str:
    return "{:.5f}_{:.5f}_{:.5f}_{:.5f}__{}_{}".format(*bbox, start, end).replace("-", "m")


def photos_from_response(photos: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Converts Flickr search results (with EXTRAS) into the standard photo table.

    Args:
        photos: The `photo` list from a search response.

    Returns:
        pd.DataFrame: Columns FLICKR_PHOTO_COLUMNS. Flickr's space-separated tags become
        comma-separated like the YFCC table.
    """
    if not photos:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in FLICKR_PHOTO_COLUMNS})
    raw = pd.DataFrame(photos)

    def col(name: str, default: Any = "") -> pd.Series:
        return raw[name] if name in raw else pd.Series(default, index=raw.index)

    return pd.DataFrame({
        "photo_id": col("id").astype(str),
        "user_id": col("owner").astype(str),
        "lon": pd.to_numeric(col("longitude"), errors="coerce"),
        "lat": pd.to_numeric(col("latitude"), errors="coerce"),
        "accuracy": pd.to_numeric(col("accuracy"), errors="coerce").fillna(0).astype("int16"),
        "taken_at": pd.to_datetime(col("datetaken"), errors="coerce", format="ISO8601"),
        "tags": col("tags").fillna("").astype(str).str.split().str.join(","),
        "uploaded_at": pd.to_datetime(pd.to_numeric(col("dateupload"), errors="coerce"), unit="s", utc=True),
        "owner_name": col("ownername").fillna("").astype(str),
        "title": col("title").fillna("").astype(str),
        "license": col("license").fillna("").astype(str),
        "thumbnail_url": col("url_q").fillna("").astype(str),
        "page_url": "https://www.flickr.com/photos/" + col("owner").astype(str) + "/" + col("id").astype(str),
    })


class FlickrFetcher:
    """
    Fetches every geotagged photo in a region and upload-date range, splitting queries that
    exceed Flickr's result cap.

    Args:
        client: A `FlickrClient` (or any object with a compatible `search` method).
        cache_dir: Directory for per-window Parquet parts (enables resuming).
        per_page: Results per page (Flickr maximum 500).
        min_window_seconds: Smallest time window before splitting in space instead.
        min_bbox_degrees: Smallest bbox side before giving up on splitting (results truncated).
        licenses: Optional comma-separated Flickr license ids to restrict results to.
        result_cap: Maximum distinct results Flickr returns per query.
    """

    def __init__(
        self,
        client: SearchClient,
        cache_dir: str,
        per_page: int = MAX_PER_PAGE,
        min_window_seconds: int = 86_400,
        min_bbox_degrees: float = 0.01,
        licenses: str | None = None,
        result_cap: int = RESULT_CAP,
    ) -> None:
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.per_page = min(per_page, MAX_PER_PAGE)
        self.min_window_seconds = min_window_seconds
        self.min_bbox_degrees = min_bbox_degrees
        self.licenses = licenses
        self.result_cap = result_cap
        self.truncated_windows: list[str] = []

    def _params(self, bbox: BBox, start: int, end: int, page: int) -> dict[str, str]:
        params = {
            "bbox": ",".join(f"{v:.6f}" for v in bbox),
            "min_upload_date": str(start),
            "max_upload_date": str(end - 1),  # inclusive on Flickr's side; windows are [start, end)
            "has_geo": "1",
            "content_type": "1",
            "media": "photos",
            "extras": EXTRAS,
            "per_page": str(self.per_page),
            "page": str(page),
            "sort": "date-posted-asc",
        }
        if self.licenses:
            params["license"] = self.licenses
        return params

    def _fetch_window(self, bbox: BBox, start: int, end: int, first: dict[str, Any]) -> pd.DataFrame:
        frames = [photos_from_response(first.get("photo", []))]
        pages = min(int(first.get("pages", 1) or 1), math.ceil(self.result_cap / self.per_page))
        for page in range(2, pages + 1):
            resp = self.client.search(self._params(bbox, start, end, page))
            frames.append(photos_from_response(resp.get("photo", [])))
        return pd.concat(frames, ignore_index=True)

    def fetch(
        self,
        bbox: BBox,
        start: datetime,
        end: datetime,
        progress: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """
        Fetches all photos uploaded in [start, end) within the bbox.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat).
            start: Inclusive upload-date start (naive datetimes are treated as UTC).
            end: Exclusive upload-date end.
            progress: Optional callback receiving one status line per finished window.

        Returns:
            pd.DataFrame: Deduplicated photos inside the bbox (FLICKR_PHOTO_COLUMNS).
        """
        to_ts = lambda d: int((d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stack: list[tuple[BBox, int, int]] = [(bbox, to_ts(start), to_ts(end))]

        while stack:
            bb, s, e = stack.pop()
            key = _window_key(bb, s, e)
            part = self.cache_dir / f"{key}.parquet"
            marker = self.cache_dir / f"{key}.split"
            if part.exists():
                continue
            if marker.exists():  # split on an earlier run: go straight to the pieces
                stack.extend(self._children(bb, s, e, marker.read_text(encoding="utf-8").strip()))
                continue

            first = self.client.search(self._params(bb, s, e, page=1))
            total = int(first.get("total", 0) or 0)

            if total > self.result_cap:
                if e - s > self.min_window_seconds:
                    how = "time"
                elif min(bb[2] - bb[0], bb[3] - bb[1]) > self.min_bbox_degrees:
                    how = "space"
                else:
                    how = ""
                if how:
                    marker.write_text(how, encoding="utf-8")
                    stack.extend(self._children(bb, s, e, how))
                    continue
                self.truncated_windows.append(key)
                logger.warning("Window %s has %d photos; only the first %d are retrievable.", key, total, self.result_cap)

            frame = self._fetch_window(bb, s, e, first) if total else photos_from_response([])
            tmp = part.with_suffix(".tmp")
            frame.to_parquet(tmp, index=False)
            os.replace(tmp, part)  # atomic: a crash never leaves a half-written part
            if progress:
                day = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d")
                progress(f"{day(s)} .. {day(e)}  bbox={bb[0]:.2f},{bb[1]:.2f},{bb[2]:.2f},{bb[3]:.2f}  {len(frame):,} photos")

        photos = self.load_cache(bbox)
        uploaded = photos["uploaded_at"]
        keep = (uploaded >= pd.Timestamp(to_ts(start), unit="s", tz="UTC")) & (
            uploaded < pd.Timestamp(to_ts(end), unit="s", tz="UTC")
        )
        return photos[keep.to_numpy()].reset_index(drop=True)

    @staticmethod
    def _children(bbox: BBox, start: int, end: int, how: str) -> list[tuple[BBox, int, int]]:
        """Sub-windows of a split window, ordered so the stack processes them oldest / south-west first."""
        if how == "time":
            mid = start + (end - start) // 2
            return [(bbox, mid, end), (bbox, start, mid)]
        return [(q, start, end) for q in reversed(_split_bbox(bbox))]

    def load_cache(self, bbox: BBox | None = None) -> pd.DataFrame:
        """
        Merges all cached parts into one deduplicated photo table.

        Args:
            bbox: Optional bbox filter (Flickr bbox matching is approximate at the edges).

        Returns:
            pd.DataFrame: Deduplicated photos (FLICKR_PHOTO_COLUMNS).
        """
        parts = sorted(self.cache_dir.glob("*.parquet"))
        frames = [pd.read_parquet(p) for p in parts]
        frames = [f for f in frames if len(f)]
        if not frames:
            return photos_from_response([])
        photos = pd.concat(frames, ignore_index=True).drop_duplicates("photo_id")
        photos = photos[photos["lon"].notna() & photos["lat"].notna() & ~((photos["lon"] == 0) & (photos["lat"] == 0))]
        if bbox is not None:
            photos = photos[in_bbox(photos["lon"].to_numpy(), photos["lat"].to_numpy(), bbox)]
        return photos.reset_index(drop=True)


def filter_accuracy(photos: pd.DataFrame, min_accuracy: int) -> pd.DataFrame:
    """
    Keeps photos whose geotag accuracy is at least `min_accuracy` (1 = world ... 16 = street).

    Args:
        photos: Photo table with an `accuracy` column.
        min_accuracy: Minimum accuracy; 0 disables the filter.

    Returns:
        pd.DataFrame: Filtered photos.
    """
    if min_accuracy <= 0:
        return photos
    return photos[np.asarray(photos["accuracy"]) >= min_accuracy].reset_index(drop=True)
