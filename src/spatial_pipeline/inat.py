"""
iNaturalist fetcher: photographed wildlife and plant observations as a photo table.

Each observation becomes one row. The observer counts as the photographer and the
species becomes the tags, so the hotspot engine answers "where do people photograph
elk / bighorn sheep / columbines?".

Design notes:

- No API key is needed. iNaturalist asks clients to stay around 1 request per second
  and ~10,000 requests per day, so requests are throttled and a run can stop after
  `max_requests` and resume later.
- Plain paging stops at 10,000 results, so the fetcher pages by observation id
  (`order_by=id`, `id_above=<last id>`), which has no cap.
- Observations with obscured coordinates (user-chosen privacy, or threatened species)
  are excluded: their public location is randomized within a large cell and would
  create false hot spots.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ._http import ThrottledJSONClient
from .geo import BBox

API_URL = "https://api.inaturalist.org/v1/observations"
MAX_PER_PAGE = 200

# Photo licenses that allow reuse with attribution (excludes "all rights reserved").
OPEN_PHOTO_LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc-by-nd,cc-by-nc-nd"

# iNaturalist "iconic taxa" -> plain-English group tag.
ICONIC_GROUPS = {
    "Aves": "birds", "Mammalia": "mammals", "Reptilia": "reptiles", "Amphibia": "amphibians",
    "Actinopterygii": "fish", "Insecta": "insects", "Arachnida": "arachnids", "Mollusca": "mollusks",
    "Plantae": "plants", "Fungi": "fungi", "Chromista": "chromista", "Protozoa": "protozoa",
    "Animalia": "animals",
}

INAT_PHOTO_COLUMNS = [
    "photo_id", "user_id", "lon", "lat", "accuracy", "accuracy_m", "taken_at", "tags",
    "uploaded_at", "owner_name", "title", "license", "thumbnail_url", "attribution",
    "taxon_id", "taxon_name", "taxon_rank", "iconic_taxon", "quality_grade",
]


class INatError(RuntimeError):
    """Raised when the iNaturalist API returns an error or the cache does not match the query."""


class ObservationClient(Protocol):
    def search(self, params: dict[str, str]) -> dict[str, Any]: ...


class INatClient(ThrottledJSONClient):
    """
    Throttled client for the iNaturalist `/v1/observations` endpoint.

    Args:
        min_interval: Minimum seconds between requests (iNaturalist asks for about 1 per second).
        max_retries: Retries for network errors, HTTP 429 and 5xx responses.
        timeout: Per-request timeout in seconds.
        sleep: Sleep function (injectable for tests).
        opener: Callable taking a urllib Request and returning a response (injectable for tests).
    """

    def __init__(
        self,
        min_interval: float = 1.0,
        max_retries: int = 4,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(min_interval=min_interval, max_retries=max_retries, timeout=timeout, sleep=sleep, opener=opener)

    def search(self, params: dict[str, str]) -> dict[str, Any]:
        """
        Queries observations.

        Args:
            params: Query parameters for `/v1/observations`.

        Returns:
            dict[str, Any]: Response with 'total_results' and 'results'.

        Raises:
            INatError: If the response carries an error message.
        """
        payload = self.get_json(API_URL, params)
        if "error" in payload:
            raise INatError(f"iNaturalist API error: {payload['error']}")
        return payload


def _clean_tag(value: Any) -> str:
    return str(value or "").replace(",", " ").strip().lower()


def observations_to_photos(results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Converts iNaturalist observation results into the photo table.

    Tags are the species' common name (or scientific name if it has none) plus its broad
    group ('birds', 'mammals', 'plants', ...). `taken_at` is the local wall-clock time of the
    observation (the time zone offset is dropped), matching YFCC's naive local times.
    `accuracy` (Flickr's 1-16 scale) is 0; iNaturalist's accuracy in meters is `accuracy_m`.

    Args:
        results: The `results` list of an observations response.

    Returns:
        pd.DataFrame: Columns INAT_PHOTO_COLUMNS, excluding obscured observations and
        observations without a photo or coordinates.
    """
    rows = []
    for obs in results:
        coords = (obs.get("geojson") or {}).get("coordinates")
        photos = obs.get("photos") or []
        if obs.get("obscured") or not coords or not photos:
            continue
        taxon = obs.get("taxon") or {}
        user = obs.get("user") or {}
        name = taxon.get("name") or ""
        common = taxon.get("preferred_common_name") or name
        group = ICONIC_GROUPS.get(taxon.get("iconic_taxon_name") or "", "")
        tags = [t for t in (_clean_tag(common), group) if t]
        observed = obs.get("time_observed_at") or obs.get("observed_on") or ""
        photo = photos[0]
        rows.append({
            "photo_id": str(obs["id"]),
            "user_id": str(user.get("id", "")),
            "lon": coords[0],
            "lat": coords[1],
            "accuracy": 0,
            "accuracy_m": obs.get("positional_accuracy"),
            "taken_at": str(observed)[:19],
            "tags": ",".join(dict.fromkeys(tags)),
            "uploaded_at": obs.get("created_at"),
            "owner_name": user.get("login") or "",
            "title": f"{common} ({name})" if common != name else name,
            "license": photo.get("license_code") or "",
            "thumbnail_url": photo.get("url") or "",
            "attribution": photo.get("attribution") or "",
            "taxon_id": taxon.get("id"),
            "taxon_name": name,
            "taxon_rank": taxon.get("rank") or "",
            "iconic_taxon": taxon.get("iconic_taxon_name") or "",
            "quality_grade": obs.get("quality_grade") or "",
        })

    if not rows:
        empty = pd.DataFrame({c: pd.Series(dtype="object") for c in INAT_PHOTO_COLUMNS})
        return empty.astype({"lon": "float64", "lat": "float64", "accuracy": "int16", "accuracy_m": "float64"})

    frame = pd.DataFrame(rows, columns=INAT_PHOTO_COLUMNS)
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce")
    frame["accuracy"] = frame["accuracy"].astype("int16")
    frame["accuracy_m"] = pd.to_numeric(frame["accuracy_m"], errors="coerce").astype("float64")
    frame["taken_at"] = pd.to_datetime(frame["taken_at"].replace("", None), errors="coerce", format="ISO8601")
    frame["uploaded_at"] = pd.to_datetime(frame["uploaded_at"], errors="coerce", utc=True, format="ISO8601")
    frame["taxon_id"] = pd.to_numeric(frame["taxon_id"], errors="coerce").astype("Int64")
    return frame


class INatFetcher:
    """
    Fetches every matching observation in a bbox by paging on observation id, with a
    resumable on-disk cache.

    Args:
        client: An `INatClient` (or any object with a compatible `search` method).
        cache_dir: Directory for per-page Parquet parts and fetch state.
        per_page: Results per request (iNaturalist maximum 200).
    """

    def __init__(self, client: ObservationClient, cache_dir: str, per_page: int = MAX_PER_PAGE) -> None:
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.per_page = min(per_page, MAX_PER_PAGE)
        self.complete = False
        self.requests_this_run = 0

    @staticmethod
    def build_query(
        bbox: BBox,
        quality_grade: str | None = "research",
        open_licenses: bool = True,
        observed_from: str | None = None,
        observed_to: str | None = None,
    ) -> dict[str, str]:
        """
        Builds the fixed part of the observations query (everything except paging).

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat).
            quality_grade: 'research' (community-verified IDs), 'needs_id', 'casual', or None for all.
            open_licenses: Only observations whose photos carry a Creative Commons license.
            observed_from: Optional first observation date (YYYY-MM-DD).
            observed_to: Optional last observation date (YYYY-MM-DD).

        Returns:
            dict[str, str]: Query parameters.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        query = {
            "swlat": f"{min_lat:.6f}", "swlng": f"{min_lon:.6f}",
            "nelat": f"{max_lat:.6f}", "nelng": f"{max_lon:.6f}",
            "photos": "true",
            "geoprivacy": "open",
            "taxon_geoprivacy": "open",
            "order_by": "id",
            "order": "asc",
        }
        if quality_grade:
            query["quality_grade"] = quality_grade
        if open_licenses:
            query["photo_license"] = OPEN_PHOTO_LICENSES
        if observed_from:
            query["d1"] = observed_from
        if observed_to:
            query["d2"] = observed_to
        return query

    def _load_state(self, query: dict[str, str]) -> dict[str, Any]:
        state_path = self.cache_dir / "state.json"
        if not state_path.exists():
            return {"query": query, "last_id": 0, "complete": False}
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("query") != query:
            raise INatError(
                f"The cache in {self.cache_dir} was created for a different query. "
                "Use a new --cache-dir or delete the old one."
            )
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        tmp = self.cache_dir / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.cache_dir / "state.json")

    def fetch(
        self,
        query: dict[str, str],
        max_requests: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> pd.DataFrame:
        """
        Fetches all observations matching `query`, resuming from the cache.

        Args:
            query: Output of `build_query`.
            max_requests: Stop after this many requests in this run (resume later). None = no limit.
            progress: Optional callback receiving one status line per page.

        Returns:
            pd.DataFrame: All cached observations so far (check `complete` to see if the fetch finished).
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        state = self._load_state(query)
        self.complete = bool(state["complete"])
        fetched = 0

        while not self.complete:
            if max_requests is not None and self.requests_this_run >= max_requests:
                break
            params = {**query, "per_page": str(self.per_page), "id_above": str(state["last_id"])}
            response = self.client.search(params)
            self.requests_this_run += 1
            results = response.get("results") or []

            if results:
                ids = [int(r["id"]) for r in results]
                frame = observations_to_photos(results)
                part = self.cache_dir / f"{min(ids):012d}_{max(ids):012d}.parquet"
                tmp = part.with_suffix(".tmp")
                frame.to_parquet(tmp, index=False)
                tmp.replace(part)
                state["last_id"] = max(ids)
                fetched += len(frame)

            if len(results) < self.per_page:
                state["complete"] = True
                self.complete = True
            self._save_state(state)

            if progress:
                remaining = max(int(response.get("total_results", 0) or 0) - len(results), 0)
                progress(f"up to id {state['last_id']:,}: +{len(results)} observations, ~{remaining:,} remaining")

        return self.load_cache()

    def load_cache(self) -> pd.DataFrame:
        """
        Merges all cached pages into one deduplicated photo table.

        Returns:
            pd.DataFrame: Observations (INAT_PHOTO_COLUMNS).
        """
        frames = [pd.read_parquet(p) for p in sorted(self.cache_dir.glob("*.parquet"))]
        frames = [f for f in frames if len(f)]
        if not frames:
            return observations_to_photos([])
        return pd.concat(frames, ignore_index=True).drop_duplicates("photo_id").reset_index(drop=True)


def filter_positional_accuracy(photos: pd.DataFrame, max_accuracy_m: float | None) -> pd.DataFrame:
    """
    Drops observations whose reported location uncertainty exceeds `max_accuracy_m` meters.
    Observations without a reported accuracy are kept.

    Args:
        photos: Table with an `accuracy_m` column.
        max_accuracy_m: Maximum uncertainty in meters; None disables the filter.

    Returns:
        pd.DataFrame: Filtered table.
    """
    if max_accuracy_m is None:
        return photos
    acc = photos["accuracy_m"].to_numpy(dtype=np.float64)
    return photos[np.isnan(acc) | (acc <= max_accuracy_m)].reset_index(drop=True)
