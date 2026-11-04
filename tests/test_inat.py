from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from spatial_pipeline import inat as inat_mod
from spatial_pipeline.cli import app
from spatial_pipeline.geo import COLORADO_BBOX, DENVER_METRO_BBOX
from spatial_pipeline.inat import (
    INatClient,
    INatError,
    INatFetcher,
    filter_positional_accuracy,
    observations_to_photos,
)

SPECIES = [
    # (lon, lat, common name, scientific name, iconic taxon)
    (-105.5850, 40.3800, "elk", "Cervus canadensis", "Mammalia"),  # Rocky Mountain NP
    (-105.2057, 39.6654, "rock squirrel", "Otospermophilus variegatus", "Mammalia"),  # Red Rocks
    (-106.9406, 39.0986, "quaking aspen", "Populus tremuloides", "Plantae"),  # Maroon Bells
    (-104.8786, 38.8784, "western scrub-jay", "Aphelocoma woodhouseii", "Aves"),  # Garden of the Gods
]


def make_observations(n: int = 1500, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    observations = []
    for i in range(n):
        lon0, lat0, common, name, iconic = SPECIES[int(rng.integers(0, len(SPECIES)))]
        user = int(rng.integers(1, 300))
        observations.append({
            "id": 1000 + i * 3,
            "geojson": {"type": "Point", "coordinates": [lon0 + rng.normal(0, 0.003), lat0 + rng.normal(0, 0.003)]},
            "obscured": False,
            "geoprivacy": None,
            "taxon_geoprivacy": None,
            "positional_accuracy": [None, 5, 40, 3000][i % 4],
            "quality_grade": "research" if i % 5 else "needs_id",
            "time_observed_at": "2021-09-18T18:42:10-06:00",
            "observed_on": "2021-09-18",
            "created_at": "2021-09-19T08:00:00-06:00",
            "user": {"id": user, "login": f"observer{user}"},
            "taxon": {"id": 40000 + len(name), "name": name, "rank": "species",
                      "preferred_common_name": common, "iconic_taxon_name": iconic},
            "photos": [{"id": 9_000_000 + i, "license_code": "cc-by" if i % 3 else None,
                        "url": f"https://static.inaturalist.org/photos/{9_000_000 + i}/square.jpg",
                        "attribution": f"(c) observer{user}, some rights reserved (CC BY)"}],
        })
    return observations


class FakeINatAPI:
    """Stand-in for /v1/observations: bbox, quality, license, geoprivacy filters and id_above paging."""

    def __init__(self, observations: list[dict], fail_after: int | None = None):
        self.observations = sorted(observations, key=lambda o: o["id"])
        self.fail_after = fail_after
        self.calls = 0

    def matching(self, params: dict[str, str]) -> list[dict]:
        out = []
        for o in self.observations:
            lon, lat = o["geojson"]["coordinates"]
            if not (float(params["swlng"]) <= lon <= float(params["nelng"]) and float(params["swlat"]) <= lat <= float(params["nelat"])):
                continue
            if "quality_grade" in params and o["quality_grade"] != params["quality_grade"]:
                continue
            if "photo_license" in params and (o["photos"][0]["license_code"] or "") not in params["photo_license"].split(","):
                continue
            if params.get("geoprivacy") == "open" and o["geoprivacy"] in ("obscured", "private"):
                continue
            out.append(o)
        return out

    def search(self, params: dict[str, str]) -> dict:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise INatError("iNaturalist API error: Service Unavailable")
        assert params["order_by"] == "id" and params["order"] == "asc" and "page" not in params
        hits = [o for o in self.matching(params) if o["id"] > int(params["id_above"])]
        per_page = int(params["per_page"])
        return {"total_results": len(hits), "page": 1, "per_page": per_page, "results": hits[:per_page]}


def query_for(bbox=COLORADO_BBOX, **kw):
    return INatFetcher.build_query(bbox, **kw)


# --- mapping ---------------------------------------------------------------

def test_observations_to_photos_mapping():
    base = make_observations(1)[0]
    no_common = json.loads(json.dumps(base))
    no_common["id"] += 1
    no_common["taxon"]["preferred_common_name"] = None
    comma = json.loads(json.dumps(base))
    comma["id"] += 2
    comma["taxon"]["preferred_common_name"] = "columbine, colorado blue"
    obscured = dict(base, id=base["id"] + 3, obscured=True)
    no_photo = dict(base, id=base["id"] + 4, photos=[])

    frame = observations_to_photos([base, no_common, comma, obscured, no_photo])

    assert frame["photo_id"].tolist() == [str(base["id"] + k) for k in (0, 1, 2)]
    assert frame.loc[0, "tags"] in {"elk,mammals", "rock squirrel,mammals", "quaking aspen,plants", "western scrub-jay,birds"}
    assert frame.loc[1, "tags"].split(",")[0] == base["taxon"]["name"].lower()
    assert frame.loc[2, "tags"].split(",")[0] == "columbine  colorado blue"
    assert frame.loc[0, "taken_at"] == pd.Timestamp("2021-09-18 18:42:10")  # local wall-clock time
    assert frame.loc[0, "uploaded_at"] == pd.Timestamp("2021-09-19 14:00:00", tz="UTC")
    assert frame.loc[0, "owner_name"].startswith("observer")
    assert frame.loc[0, "thumbnail_url"].endswith("square.jpg")
    assert len(observations_to_photos([])) == 0


def test_filter_positional_accuracy_keeps_unknown():
    frame = observations_to_photos(make_observations(8))
    kept = filter_positional_accuracy(frame, 1000.0)
    assert kept["accuracy_m"].isna().sum() == frame["accuracy_m"].isna().sum()
    assert (kept["accuracy_m"].dropna() <= 1000).all()
    assert len(filter_positional_accuracy(frame, None)) == len(frame)


# --- fetcher ---------------------------------------------------------------

def test_fetcher_pages_by_id_and_applies_filters(tmp_path):
    observations = make_observations(1500)
    observations[7]["geoprivacy"] = "obscured"
    observations[8]["obscured"] = True  # obscured by taxon even though geoprivacy is open
    api = FakeINatAPI(observations)
    fetcher = INatFetcher(api, str(tmp_path / "cache"), per_page=200)
    query = query_for()

    got = fetcher.fetch(query)

    expected = {str(o["id"]) for o in api.matching(query) if not o["obscured"]}
    assert set(got["photo_id"]) == expected
    assert fetcher.complete
    assert api.calls == len(api.matching(query)) // 200 + 1
    assert (got["license"] == "cc-by").all()


def test_fetcher_respects_max_requests_and_resumes(tmp_path):
    api = FakeINatAPI(make_observations(1500))
    cache = str(tmp_path / "cache")
    query = query_for(quality_grade=None, open_licenses=False)

    first = INatFetcher(api, cache, per_page=100)
    partial = first.fetch(query, max_requests=4)
    assert not first.complete and len(partial) == 400

    second = INatFetcher(api, cache, per_page=100)
    full = second.fetch(query)
    assert second.complete
    assert set(full["photo_id"]) == {str(o["id"]) for o in api.observations}

    third_api = FakeINatAPI(make_observations(1500))
    third = INatFetcher(third_api, cache, per_page=100)
    third.fetch(query)
    assert third_api.calls == 0


def test_fetcher_recovers_after_errors(tmp_path):
    cache = str(tmp_path / "cache")
    query = query_for(open_licenses=False)
    with pytest.raises(INatError):
        INatFetcher(FakeINatAPI(make_observations(900), fail_after=2), cache, per_page=100).fetch(query)
    api = FakeINatAPI(make_observations(900))
    got = INatFetcher(api, cache, per_page=100).fetch(query)
    assert set(got["photo_id"]) == {str(o["id"]) for o in api.matching(query)}


def test_fetcher_rejects_cache_from_other_query(tmp_path):
    cache = str(tmp_path / "cache")
    INatFetcher(FakeINatAPI(make_observations(50)), cache).fetch(query_for())
    with pytest.raises(INatError, match="different query"):
        INatFetcher(FakeINatAPI(make_observations(50)), cache).fetch(query_for(bbox=DENVER_METRO_BBOX))


# --- HTTP client -----------------------------------------------------------

class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_client_sends_query_and_raises_api_errors():
    seen = []

    def opener(request, timeout):
        seen.append(request)
        if len(seen) == 1:
            return FakeResponse(json.dumps({"total_results": 0, "results": []}).encode())
        return FakeResponse(json.dumps({"error": "Result window is too large", "status": 403}).encode())

    client = INatClient(min_interval=0, opener=opener, sleep=lambda s: None)
    assert client.search({"id_above": "5", "per_page": "200"})["total_results"] == 0
    assert seen[0].full_url.startswith("https://api.inaturalist.org/v1/observations?")
    assert "id_above=5" in seen[0].full_url
    assert "spatial-pipeline" in seen[0].get_header("User-agent")
    with pytest.raises(INatError, match="too large"):
        client.search({})


# --- CLI -------------------------------------------------------------------

def test_cli_fetch_inat_then_hotspots(tmp_path, monkeypatch):
    api = FakeINatAPI(make_observations(1500))
    monkeypatch.setattr(inat_mod, "INatClient", lambda min_interval=1.0: api)
    out = str(tmp_path / "inat.parquet")
    runner = CliRunner()

    result = runner.invoke(app, ["fetch-inat", "--output", out, "--cache-dir", str(tmp_path / "cache")])
    assert result.exit_code == 0, result.output
    photos = pd.read_parquet(out)
    assert len(photos) > 0 and (photos["accuracy_m"].dropna() <= 1000).all()

    result = runner.invoke(app, [
        "hotspots", "--input", out, "--output-dir", str(tmp_path / "hot"),
        "--min-label-photographers", "5", "--min-hex-photographers", "1",
    ])
    assert result.exit_code == 0, result.output
    ranking = pd.read_csv(tmp_path / "hot" / "ranking.csv").set_index("label")
    assert ranking.loc["elk", "hotspot_hexes"] > 0
    assert "mammals" in ranking.index


def test_cli_fetch_inat_partial_run(tmp_path, monkeypatch):
    api = FakeINatAPI(make_observations(1500))
    monkeypatch.setattr(inat_mod, "INatClient", lambda min_interval=1.0: api)
    result = CliRunner().invoke(app, [
        "fetch-inat", "--output", str(tmp_path / "p.parquet"), "--cache-dir", str(tmp_path / "c"),
        "--max-requests", "1", "--quality", "all", "--all-licenses",
    ])
    assert result.exit_code == 0, result.output
    assert "rerun" in result.output.lower()


def test_cli_fetch_inat_rejects_bad_quality():
    result = CliRunner().invoke(app, ["fetch-inat", "--quality", "best"])
    assert result.exit_code != 0
