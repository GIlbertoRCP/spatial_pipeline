from __future__ import annotations

import io
import json
import math
import urllib.error
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from spatial_pipeline import flickr as flickr_mod
from spatial_pipeline.cli import app
from spatial_pipeline.flickr import (
    FlickrClient,
    FlickrError,
    FlickrFetcher,
    filter_accuracy,
    photos_from_response,
)
from spatial_pipeline.geo import COLORADO_BBOX
from tests.yfcc_fixture import LANDMARKS

T0 = datetime(2016, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2020, 1, 1, tzinfo=timezone.utc)


class FakeFlickrAPI:
    """In-memory stand-in for flickr.photos.search: bbox + upload-date filters, paging, result cap."""

    def __init__(self, photos: pd.DataFrame, result_cap: int = 4000, fail_after: int | None = None):
        self.photos = photos
        self.result_cap = result_cap
        self.fail_after = fail_after
        self.calls = 0

    @property
    def requests_made(self) -> int:
        return self.calls

    def search(self, params: dict[str, str]) -> dict:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise FlickrError("Flickr API error 105: Service currently unavailable")
        assert params["has_geo"] == "1" and "min_upload_date" in params and "max_upload_date" in params

        min_lon, min_lat, max_lon, max_lat = (float(v) for v in params["bbox"].split(","))
        p = self.photos
        mask = (
            (p["lon"] >= min_lon) & (p["lon"] <= max_lon) & (p["lat"] >= min_lat) & (p["lat"] <= max_lat)
            & (p["dateupload"] >= int(params["min_upload_date"])) & (p["dateupload"] <= int(params["max_upload_date"]))
        )
        if "license" in params:
            mask &= p["license"].astype(str).isin(params["license"].split(","))
        hits = p[mask].sort_values("dateupload")
        per_page = int(params["per_page"])
        page = int(params["page"])
        # Like Flickr: only the first `result_cap` results are reachable; later pages repeat.
        reachable = hits.head(self.result_cap)
        last_page = max(1, math.ceil(len(reachable) / per_page))
        rows = reachable.iloc[(min(page, last_page) - 1) * per_page: min(page, last_page) * per_page]
        return {
            "page": page,
            "pages": max(1, math.ceil(len(hits) / per_page)),
            "perpage": per_page,
            "total": str(len(hits)),
            "photo": [
                {
                    "id": str(r.id), "owner": r.owner, "title": f"photo {r.id}",
                    "latitude": f"{r.lat:.6f}", "longitude": f"{r.lon:.6f}", "accuracy": str(r.accuracy),
                    "tags": r.tags, "datetaken": "2017-08-01 18:30:00", "dateupload": str(r.dateupload),
                    "ownername": r.owner.split("@")[0], "license": str(r.license),
                    "url_q": f"https://live.staticflickr.com/1/{r.id}_abc_q.jpg",
                }
                for r in rows.itertuples()
            ],
        }


def make_photos(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spots = list(LANDMARKS.values())
    which = rng.integers(0, len(spots), size=n)
    lon = np.array([spots[i][0] for i in which]) + rng.normal(0, 0.003, n)
    lat = np.array([spots[i][1] for i in which]) + rng.normal(0, 0.003, n)
    tags = [" ".join(t.replace(" ", "") for t in spots[i][2]) for i in which]
    upload = rng.integers(int(T0.timestamp()), int(T1.timestamp()), size=n)
    return pd.DataFrame({
        "id": np.arange(10_000, 10_000 + n), "owner": [f"{u}@N0{u % 7}" for u in rng.integers(0, 400, n)],
        "lon": lon, "lat": lat, "accuracy": rng.choice([11, 14, 16], n), "tags": tags,
        "dateupload": upload, "license": rng.choice([0, 4, 9], n),
    })


def expected_ids(photos: pd.DataFrame) -> set[str]:
    return set(photos["id"].astype(str))


# --- fetcher ---------------------------------------------------------------

def test_fetcher_splits_windows_to_recover_everything(tmp_path):
    photos = make_photos(3000)
    api = FakeFlickrAPI(photos, result_cap=300)
    fetcher = FlickrFetcher(api, str(tmp_path / "cache"), per_page=100, result_cap=300)

    got = fetcher.fetch(COLORADO_BBOX, T0, T1)

    assert set(got["photo_id"]) == expected_ids(photos)
    assert got["photo_id"].is_unique
    assert fetcher.truncated_windows == []
    assert set(got.columns) >= {"photo_id", "user_id", "lon", "lat", "accuracy", "taken_at", "tags"}


def test_fetcher_splits_in_space_when_time_cannot_split(tmp_path):
    photos = make_photos(1200)
    photos["dateupload"] = int(T0.timestamp()) + 3600  # everything uploaded in the same hour
    api = FakeFlickrAPI(photos, result_cap=400)
    fetcher = FlickrFetcher(api, str(tmp_path / "cache"), per_page=100, result_cap=400)

    got = fetcher.fetch(COLORADO_BBOX, T0, T1)

    assert set(got["photo_id"]) == expected_ids(photos)
    assert fetcher.truncated_windows == []


def test_fetcher_reports_truncation_when_unsplittable(tmp_path):
    photos = make_photos(500)
    photos["dateupload"] = int(T0.timestamp()) + 3600
    photos["lon"], photos["lat"] = -105.2057, 39.6654  # same instant, same spot
    api = FakeFlickrAPI(photos, result_cap=200)
    fetcher = FlickrFetcher(api, str(tmp_path / "cache"), per_page=100, result_cap=200)

    got = fetcher.fetch(COLORADO_BBOX, T0, T1)

    assert len(got) == 200
    assert len(fetcher.truncated_windows) == 1


def test_fetcher_resumes_from_cache(tmp_path):
    photos = make_photos(2000)
    cache = str(tmp_path / "cache")

    # First run dies partway through.
    failing = FakeFlickrAPI(photos, result_cap=300, fail_after=15)
    with pytest.raises(FlickrError):
        FlickrFetcher(failing, cache, per_page=100, result_cap=300).fetch(COLORADO_BBOX, T0, T1)

    # Second run finishes the job.
    api = FakeFlickrAPI(photos, result_cap=300)
    got = FlickrFetcher(api, cache, per_page=100, result_cap=300).fetch(COLORADO_BBOX, T0, T1)
    assert set(got["photo_id"]) == expected_ids(photos)

    # A third run is fully cached: no requests at all.
    again = FakeFlickrAPI(photos, result_cap=300)
    got2 = FlickrFetcher(again, cache, per_page=100, result_cap=300).fetch(COLORADO_BBOX, T0, T1)
    assert again.calls == 0
    assert set(got2["photo_id"]) == expected_ids(photos)


def test_fetcher_filters_licenses_and_date_range(tmp_path):
    photos = make_photos(800)
    api = FakeFlickrAPI(photos)
    mid = datetime(2018, 1, 1, tzinfo=timezone.utc)
    fetcher = FlickrFetcher(api, str(tmp_path / "cache"), licenses="4,9")

    got = fetcher.fetch(COLORADO_BBOX, T0, mid)

    want = photos[(photos["license"] != 0) & (photos["dateupload"] < int(mid.timestamp()))]
    assert set(got["photo_id"]) == expected_ids(want)


def test_photos_from_response_and_accuracy_filter():
    frame = photos_from_response([
        {"id": "1", "owner": "a@N00", "latitude": "39.1", "longitude": "-106.9", "accuracy": "16",
         "tags": "maroonbells aspen", "datetaken": "2019-09-20 07:01:00", "dateupload": "1568962860"},
        {"id": "2", "owner": "b@N00", "latitude": 39.7, "longitude": -105.2, "accuracy": 11, "tags": ""},
    ])
    assert frame["tags"].tolist() == ["maroonbells,aspen", ""]
    assert frame["lat"].tolist() == [39.1, 39.7]
    assert frame.loc[0, "uploaded_at"] == pd.Timestamp(1568962860, unit="s", tz="UTC")
    assert frame.loc[0, "taken_at"] == pd.Timestamp("2019-09-20 07:01:00")
    assert frame["thumbnail_url"].tolist() == ["", ""]
    assert filter_accuracy(frame, 12)["photo_id"].tolist() == ["1"]
    assert len(photos_from_response([])) == 0


# --- HTTP client -----------------------------------------------------------

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_client_builds_request_and_parses_response():
    seen = []

    def opener(request, timeout):
        seen.append(request)
        return FakeResponse(json.dumps({"stat": "ok", "photos": {"page": 1, "pages": 1, "total": "0", "photo": []}}).encode())

    client = FlickrClient("secret-key", min_interval=0, opener=opener, sleep=lambda s: None)
    result = client.search({"bbox": "1,2,3,4", "page": "1"})

    assert result["total"] == "0"
    url = seen[0].full_url
    assert "method=flickr.photos.search" in url and "api_key=secret-key" in url and "format=json" in url
    assert "spatial-pipeline" in seen[0].get_header("User-agent")


def test_client_retries_transient_errors_then_raises_api_errors():
    sleeps: list[float] = []
    responses = [
        urllib.error.HTTPError("u", 503, "unavailable", {}, None),
        urllib.error.URLError("connection reset"),
        FakeResponse(json.dumps({"stat": "fail", "code": 100, "message": "Invalid API Key"}).encode()),
    ]

    def opener(request, timeout):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = FlickrClient("bad-key", min_interval=0, opener=opener, sleep=sleeps.append)
    with pytest.raises(FlickrError, match="Invalid API Key"):
        client.search({})
    assert client.requests_made == 3
    assert sleeps == [2.0, 4.0]


def test_client_does_not_retry_client_errors():
    def opener(request, timeout):
        raise urllib.error.HTTPError("u", 403, "forbidden", {}, None)

    client = FlickrClient("k", min_interval=0, opener=opener, sleep=lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        client.search({})
    assert client.requests_made == 1


def test_client_throttles_requests():
    sleeps: list[float] = []

    def opener(request, timeout):
        return FakeResponse(json.dumps({"stat": "ok", "photos": {}}).encode())

    client = FlickrClient("k", min_interval=5.0, opener=opener, sleep=sleeps.append)
    client.search({})
    client.search({})
    assert len(sleeps) == 1 and 4.0 < sleeps[0] <= 5.0


def test_client_requires_key():
    with pytest.raises(ValueError):
        FlickrClient("")


# --- CLI -------------------------------------------------------------------

def test_cli_fetch_flickr_then_hotspots(tmp_path, monkeypatch):
    photos = make_photos(1500)
    api = FakeFlickrAPI(photos)
    monkeypatch.setattr(flickr_mod, "FlickrClient", lambda key, min_interval=1.0: api)
    out = str(tmp_path / "flickr.parquet")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["fetch-flickr", "--end", "2020-01-01", "--start", "2016-01-01", "--output", out,
         "--cache-dir", str(tmp_path / "cache"), "--min-accuracy", "0"],
        env={"FLICKR_API_KEY": "test-key"},
    )
    assert result.exit_code == 0, result.output
    assert "test-key" not in result.output
    assert len(pd.read_parquet(out)) == len(photos)

    result = runner.invoke(app, [
        "hotspots", "--input", out, "--output-dir", str(tmp_path / "hot"),
        "--min-label-photographers", "5", "--min-hex-photographers", "1",
    ])
    assert result.exit_code == 0, result.output
    ranking = pd.read_csv(tmp_path / "hot" / "ranking.csv")
    assert "maroonbells" in set(ranking["label"])
    assert ranking.loc[ranking["label"] == "maroonbells", "hotspot_hexes"].iloc[0] > 0


def test_cli_fetch_flickr_requires_key_and_valid_dates(tmp_path, monkeypatch):
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)
    runner = CliRunner()
    result = runner.invoke(app, ["fetch-flickr", "--end", "2020-01-01"], env={})
    assert result.exit_code != 0

    result = runner.invoke(app, ["fetch-flickr", "--start", "2020-01-01", "--end", "2019-01-01"], env={"FLICKR_API_KEY": "k"})
    assert result.exit_code != 0


def test_cli_fetch_flickr_reports_api_errors(tmp_path, monkeypatch):
    api = FakeFlickrAPI(make_photos(10), fail_after=0)
    monkeypatch.setattr(flickr_mod, "FlickrClient", lambda key, min_interval=1.0: api)
    result = CliRunner().invoke(
        app, ["fetch-flickr", "--end", "2020-01-01", "--cache-dir", str(tmp_path / "c")],
        env={"FLICKR_API_KEY": "k"},
    )
    assert result.exit_code == 1
    assert "rerun to resume" in result.output
