from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from spatial_pipeline.cli import app
from spatial_pipeline.geo import COLORADO_BBOX, parse_bbox, project_lonlat, unproject_xy
from spatial_pipeline.hotspots import compute_hotspots, load_hotspots, save_hotspots
from spatial_pipeline.photos import (
    decode_tags,
    explode_tags,
    load_photos,
    read_yfcc_metadata,
    save_photos,
)
from tests.yfcc_fixture import (
    LANDMARKS,
    PROLIFIC_USER,
    make_rows,
    write_sqlite,
    write_tsv,
)

EXPECTED_KEPT = len(make_rows()) - 5  # outside bbox, video, no geo, low accuracy, duplicate


# --- geo -------------------------------------------------------------------

def test_parse_bbox_named_and_custom():
    assert parse_bbox("Colorado") == COLORADO_BBOX
    assert parse_bbox("-105.3, 39.5, -104.6, 39.9") == (-105.3, 39.5, -104.6, 39.9)
    with pytest.raises(ValueError):
        parse_bbox("1,2,3")
    with pytest.raises(ValueError):
        parse_bbox("-104,39,-105,40")  # min_lon > max_lon


def test_projection_roundtrip_and_units():
    lon = np.array([-104.9903, -105.2057])
    lat = np.array([39.7392, 39.6654])
    x, y = project_lonlat(lon, lat)
    back_lon, back_lat = unproject_xy(x, y)
    np.testing.assert_allclose(back_lon, lon, atol=1e-9)
    np.testing.assert_allclose(back_lat, lat, atol=1e-9)
    # Planar distance in meters should match the true geodesic distance within 1%.
    from pyproj import Geod
    _, _, geodesic = Geod(ellps="WGS84").inv(lon[0], lat[0], lon[1], lat[1])
    dist = float(np.hypot(x[1] - x[0], y[1] - y[0]))
    assert abs(dist - geodesic) / geodesic < 0.01


# --- photo ingestion -------------------------------------------------------

@pytest.mark.parametrize("width", [23, 25])
def test_read_yfcc_tsv_layouts(tmp_path, width):
    path = str(tmp_path / f"yfcc_{width}.tsv")
    write_tsv(path, width=width)
    photos = read_yfcc_metadata(path, bbox=COLORADO_BBOX, chunksize=97)

    assert len(photos) == EXPECTED_KEPT
    assert photos["photo_id"].is_unique
    assert photos["lon"].between(COLORADO_BBOX[0], COLORADO_BBOX[2]).all()
    assert (photos["accuracy"] >= 12).all()
    assert not photos["tags"].str.contains("sanfrancisco|nogeo").any()
    assert pd.api.types.is_datetime64_any_dtype(photos["taken_at"])


def test_read_yfcc_bz2_and_sqlite_match_tsv(tmp_path):
    tsv = str(tmp_path / "yfcc.tsv")
    bz2 = str(tmp_path / "yfcc.tsv.bz2")
    db = str(tmp_path / "yfcc.sql")
    rows = write_tsv(tsv)
    write_tsv(bz2, rows=rows)
    write_sqlite(db, rows=rows)

    ref = read_yfcc_metadata(tsv, bbox=COLORADO_BBOX).sort_values("photo_id").reset_index(drop=True)
    for path in (bz2, db):
        other = read_yfcc_metadata(path, bbox=COLORADO_BBOX, chunksize=50).sort_values("photo_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(ref, other)


def test_read_yfcc_filters_can_be_relaxed(tmp_path):
    path = str(tmp_path / "yfcc.tsv")
    write_tsv(path)
    photos = read_yfcc_metadata(path, bbox=None, min_accuracy=0, include_videos=True)
    # Only the non-geotagged row and the duplicate are dropped.
    assert len(photos) == len(make_rows()) - 2


def test_read_yfcc_rejects_unknown_layout(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("a\tb\tc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="23 or 25"):
        read_yfcc_metadata(str(path))


def test_decode_and_explode_tags():
    decoded = decode_tags(pd.Series(["Rocky+Mountains,colorado", "", None, "caf%C3%A9"]))
    assert decoded.tolist() == ["rocky mountains,colorado", "", "", "café"]

    photos = pd.DataFrame({"tags": ["waterfall,2012,iphone,ab,waterfall,geo:lat=39", "lake"]})
    tags = explode_tags(photos)
    assert sorted(map(tuple, tags.to_numpy().tolist())) == [(0, "waterfall"), (1, "lake")]


def test_save_and_load_photos_roundtrip(tmp_path):
    src = str(tmp_path / "yfcc.tsv")
    write_tsv(src)
    photos = read_yfcc_metadata(src, bbox=COLORADO_BBOX)
    for name in ("photos.parquet", "photos.csv"):
        out = str(tmp_path / name)
        save_photos(photos, out)
        loaded = load_photos(out)
        assert len(loaded) == len(photos)
        assert set(loaded["photo_id"]) == set(photos["photo_id"])


# --- hotspots --------------------------------------------------------------

@pytest.fixture(scope="module")
def photos(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("yfcc") / "yfcc.tsv")
    write_tsv(path)
    return read_yfcc_metadata(path, bbox=COLORADO_BBOX)


def test_hotspots_count_photographers_not_photos(photos):
    result = compute_hotspots(photos, resolution=1000.0, min_label_photographers=1, min_hex_photographers=1)
    prolific = photos[photos["user_id"] == PROLIFIC_USER]
    assert len(prolific) == 300

    # The prolific user's hex has ~300 photos but a single photographer.
    top_by_photos = result.hexes.sort_values("photos", ascending=False).iloc[0]
    assert top_by_photos["photos"] >= 290
    assert top_by_photos["photographers"] == 1

    backyard = result.label_counts[result.label_counts["label"] == "backyard"]
    assert backyard["photographers"].sum() <= len(backyard)  # one photographer per hex


def test_hotspots_rank_landmark_tags_above_diffuse_tags(photos):
    result = compute_hotspots(photos, resolution=1000.0, min_label_photographers=5, min_hex_photographers=1)
    ranking = result.ranking.set_index("label")

    assert "iphone" not in ranking.index and "2012" not in ranking.index
    assert "backyard" not in ranking.index  # only one photographer
    landmark_tags = {tag for _, _, tags in LANDMARKS.values() for tag in tags}
    top = set(result.ranking.head(len(landmark_tags))["label"])
    assert len(top & landmark_tags) >= len(landmark_tags) - 2
    assert ranking.loc["maroon bells", "morans_i"] > ranking.loc["colorado", "morans_i"]
    assert ranking.loc["maroon bells", "photographers"] == 30


def test_hotspots_hex_geometry(photos):
    result = compute_hotspots(photos, resolution=1000.0, min_label_photographers=5, min_hex_photographers=1)
    lon0, lat0, _ = LANDMARKS["red_rocks"]
    nearest = result.hexes.assign(d=np.hypot(result.hexes["lon"] - lon0, result.hexes["lat"] - lat0)).nsmallest(1, "d")
    assert nearest["d"].iloc[0] < 0.01  # hex center within ~1 km of the landmark

    # Hex neighbors are exactly one resolution apart.
    adj = result.adjacency.tocoo()
    centers = result.hexes.set_index("hex_id").loc[result.stats_hex_ids, ["x", "y"]].to_numpy()
    if adj.nnz:
        d = np.hypot(*(centers[adj.row] - centers[adj.col]).T)
        np.testing.assert_allclose(d, 1000.0, rtol=1e-6)


def test_save_and_load_hotspots(tmp_path, photos):
    result = compute_hotspots(photos, resolution=2000.0, min_label_photographers=5, min_hex_photographers=1)
    save_hotspots(result, str(tmp_path / "out"), extra_meta={"input": "fixture"})
    loaded = load_hotspots(str(tmp_path / "out"))
    assert len(loaded["hexes"]) == len(result.hexes)
    assert loaded["meta"]["resolution"] == 2000.0
    assert loaded["meta"]["input"] == "fixture"
    assert list(loaded["ranking"]["label"]) == list(result.ranking["label"])


def test_hotspots_empty_input_raises():
    with pytest.raises(ValueError):
        compute_hotspots(pd.DataFrame(columns=["photo_id", "user_id", "lon", "lat", "tags"]))


# --- CLI -------------------------------------------------------------------

def test_cli_ingest_and_hotspots(tmp_path):
    runner = CliRunner()
    src = str(tmp_path / "yfcc.tsv")
    photos_path = str(tmp_path / "photos.parquet")
    out_dir = str(tmp_path / "hotspots")
    write_tsv(src)

    result = runner.invoke(app, ["ingest-yfcc", "--input", src, "--output", photos_path, "--bbox", "colorado"])
    assert result.exit_code == 0, result.output
    assert os.path.exists(photos_path)

    result = runner.invoke(app, [
        "hotspots", "--input", photos_path, "--output-dir", out_dir,
        "--min-label-photographers", "5", "--min-hex-photographers", "1", "--top-n", "5",
    ])
    assert result.exit_code == 0, result.output
    assert "Spatially Clustered Tags" in result.output
    for name in ("hexes.parquet", "label_counts.parquet", "ranking.csv", "meta.json"):
        assert os.path.exists(os.path.join(out_dir, name))


def test_hex_polygons_match_hex_size(photos):
    from spatial_pipeline.geo import project_lonlat
    from spatial_pipeline.hotspots import hex_polygons_lonlat

    result = compute_hotspots(photos, resolution=1000.0, min_label_photographers=5, min_hex_photographers=1)
    hexes = result.hexes.head(5)
    rings = hex_polygons_lonlat(hexes["x"].to_numpy(), hexes["y"].to_numpy(), 1000.0)
    assert len(rings) == 5 and all(len(r) == 6 for r in rings)
    for ring, (_, h) in zip(rings, hexes.iterrows()):
        vx, vy = project_lonlat(np.array([p[0] for p in ring]), np.array([p[1] for p in ring]))
        np.testing.assert_allclose(np.hypot(vx - h["x"], vy - h["y"]), 1000.0 / np.sqrt(3.0), rtol=1e-6)


def test_streamlit_app_renders(tmp_path, photos, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from spatial_pipeline.timing import compute_timing, save_timing

    out_dir = str(tmp_path / "hotspots")
    save_hotspots(compute_hotspots(photos, min_label_photographers=5, min_hex_photographers=1), out_dir)
    save_timing(compute_timing(photos), out_dir)
    monkeypatch.setenv("HOTSPOTS_DIR", out_dir)
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")

    at = AppTest.from_file(app_path, default_timeout=60).run()
    assert not at.exception
    assert at.metric[0].value == f"{len(photos):,}"

    at.selectbox[0].select("red rocks").run()
    assert not at.exception
    assert not at.warning
    assert any(s.value == "When to go" for s in at.subheader)
    assert any("Jul" in c.value and "19:00" in c.value for c in at.caption)  # fixture photos: July, 19:42

    at.checkbox[0].check().run()  # only significant hot spots
    assert not at.exception
    assert not at.warning
    top_hexes = at.dataframe[0].value
    assert len(top_hexes) > 0 and top_hexes["gi_z"].notna().all()


def test_local_hotspots_mark_landmark_hexes(photos):
    result = compute_hotspots(photos, resolution=1000.0, min_label_photographers=5, min_hex_photographers=1)
    local = result.local_hotspots
    assert (local["gi_z"] > 0).all() and (local["fdr_p_value"] < 0.05).all()
    assert not local["label"].isin(["colorado", "mountains"]).any()

    hexes = result.hexes.set_index("hex_id")
    for name, (lon0, lat0, tags) in LANDMARKS.items():
        hot = hexes.loc[local.loc[local["label"] == tags[0], "hex_id"]]
        assert len(hot) > 0, name
        # Every hot-spot hex for a landmark's tag lies within ~3 km of that landmark.
        assert (np.hypot(hot["lon"] - lon0, hot["lat"] - lat0) < 0.03).all(), name

    ranking = result.ranking.set_index("label")
    assert ranking.loc["red rocks", "hotspot_hexes"] > 0
    assert ranking.loc["colorado", "hotspot_hexes"] == 0
    stats = hexes.loc[result.stats_hex_ids]
    assert stats["density_gi_z"].notna().all()
    assert hexes.drop(index=result.stats_hex_ids)["density_gi_z"].isna().all()


def test_load_hotspots_without_local_file(tmp_path, photos):
    out_dir = tmp_path / "old"
    save_hotspots(compute_hotspots(photos, min_label_photographers=5, min_hex_photographers=1), str(out_dir))
    (out_dir / "local_hotspots.parquet").unlink()
    loaded = load_hotspots(str(out_dir))
    assert loaded["local_hotspots"].empty


def test_yfcc_rows_carry_thumbnails_and_credits(photos):
    assert photos["thumbnail_url"].str.match(r"^https://live\.staticflickr\.com/1/\d+_abc_q\.jpg$").all()
    assert photos["page_url"].str.startswith("http://www.flickr.com/photos/").all()
    assert (photos["license"] == "Attribution License").all()
    assert photos["owner_name"].ne("").all()


def test_sample_photos_one_per_photographer_per_group(photos):
    from spatial_pipeline.hotspots import sample_photos
    from spatial_pipeline.timing import ALL_LABEL

    result = compute_hotspots(photos, min_label_photographers=5, min_hex_photographers=1)
    samples = sample_photos(photos, result, per_group=4)

    assert set(samples["label"]) <= set(result.labels) | {ALL_LABEL}
    assert samples.groupby(["hex_id", "label"]).size().max() <= 4
    owners = photos.set_index("photo_id")["user_id"]
    users = samples.assign(user=samples["photo_id"].map(owners))
    assert not users.duplicated(["hex_id", "label", "user"]).any()
    # The prolific uploader's hex gets one sample, not 300.
    prolific_hex = result.hexes.sort_values("photos").iloc[-1]["hex_id"]
    assert len(samples[(samples["hex_id"] == prolific_hex) & (samples["label"] == ALL_LABEL)]) == 1
    # No thumbnails -> empty sample.
    assert sample_photos(photos.drop(columns="thumbnail_url"), result).empty


def test_streamlit_app_selected_spot_panel(tmp_path, photos, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from spatial_pipeline.hotspots import sample_photos, save_photo_samples

    out_dir = str(tmp_path / "hotspots")
    result = compute_hotspots(photos, min_label_photographers=5, min_hex_photographers=1)
    save_hotspots(result, out_dir)
    save_photo_samples(sample_photos(photos, result), out_dir)
    monkeypatch.setenv("HOTSPOTS_DIR", out_dir)
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")

    red_rocks_hex = int(result.local_hotspots.loc[result.local_hotspots["label"] == "red rocks", "hex_id"].iloc[0])
    at = AppTest.from_file(app_path, default_timeout=60)
    at.query_params["hex"] = str(red_rocks_hex)
    at.run()
    assert not at.exception
    assert any(s.value == "Selected spot" for s in at.subheader)
    assert any("photo-grid" in m.value and "staticflickr" in m.value for m in at.markdown)
    subjects = at.dataframe[0].value
    assert "★ red rocks" in set(subjects["subject"])  # ★ = significant hot spot for that subject

    at.selectbox[0].select("red rocks").run()
    assert not at.exception
    assert any("★ significant hot spot" in m.value for m in at.markdown)

    at.button[0].click().run()  # clear selection
    assert not at.exception
    assert not any(s.value == "Selected spot" for s in at.subheader)


SEASON_OF = {"maroon bells": 9, "red rocks": 7, "garden of the gods": 4, "denver": 12}


def seasonal(photos: pd.DataFrame) -> pd.DataFrame:
    """Fixture photos with each landmark photographed in its own month (others spread over the year)."""
    rng = np.random.default_rng(1)
    months = []
    for tags in photos["tags"]:
        hit = next((m for t, m in SEASON_OF.items() if t in tags.split(",")), None)
        months.append(hit if hit is not None else int(rng.integers(1, 13)))
    taken = [pd.Timestamp(2012, m, 14, 18, 30) for m in months]
    return photos.assign(taken_at=taken)


def test_subject_info_uses_taxonomy_when_present():
    from spatial_pipeline.hotspots import subject_info

    photos = pd.DataFrame({
        "tags": ["elk,mammals", "elk,mammals", "quaking aspen,plants", "skyline"],
        "iconic_taxon": ["Mammalia", "Mammalia", "Plantae", ""],
        "taxon_name": ["Cervus canadensis", "Cervus canadensis", "Populus tremuloides", ""],
    })
    info = subject_info(photos, ["elk", "mammals", "quaking aspen", "skyline"]).set_index("label")
    assert info.loc["elk", "group"] == "mammals" and info.loc["elk", "scientific_name"] == "Cervus canadensis"
    assert info.loc["mammals", "group"] == "mammals" and info.loc["mammals", "scientific_name"] == ""
    assert info.loc["skyline", "group"] == "" and info.loc["skyline", "scientific_name"] == ""
    assert subject_info(photos.drop(columns=["iconic_taxon"]), ["elk"])["group"].tolist() == [""]


def test_monthly_label_counts(photos):
    from spatial_pipeline.hotspots import monthly_label_counts
    from spatial_pipeline.timing import ALL_LABEL

    p = seasonal(photos)
    result = compute_hotspots(p, min_label_photographers=5, min_hex_photographers=1)
    monthly = monthly_label_counts(p, result)

    bells = monthly[monthly["label"] == "maroon bells"]
    assert set(bells["month"]) == {9}
    assert bells["photographers"].sum() == result.label_counts.query("label == 'maroon bells'")["photographers"].sum()
    overall = monthly[monthly["label"] == ALL_LABEL]
    assert overall.groupby("hex_id")["photographers"].sum().ge(1).all()
    prolific_hex = result.hexes.sort_values("photos").iloc[-1]["hex_id"]
    assert (overall[overall["hex_id"] == prolific_hex]["photographers"] == 1).all()  # one person, once per month
    assert monthly_label_counts(p.assign(taken_at=pd.NaT), result).empty


def test_streamlit_app_month_and_group_filters(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from spatial_pipeline.hotspots import (
        monthly_label_counts,
        sample_photos,
        save_photo_samples,
        subject_info,
    )
    from spatial_pipeline.timing import compute_timing, save_timing

    src = str(tmp_path / "yfcc.tsv")
    write_tsv(src)
    p = seasonal(read_yfcc_metadata(src, bbox=COLORADO_BBOX))
    result = compute_hotspots(p, min_label_photographers=5, min_hex_photographers=1)
    out_dir = str(tmp_path / "hotspots")
    save_hotspots(result, out_dir)
    save_timing(compute_timing(p), out_dir)
    save_photo_samples(sample_photos(p, result), out_dir)
    subject_info(p, result.labels).to_parquet(os.path.join(out_dir, "subjects.parquet"), index=False)
    monthly_label_counts(p, result).to_parquet(os.path.join(out_dir, "monthly_counts.parquet"), index=False)
    monkeypatch.setenv("HOTSPOTS_DIR", out_dir)
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")

    at = AppTest.from_file(app_path, default_timeout=60).run()
    assert not at.exception

    at.select_slider(key="when").set_value("Sep").run()
    assert not at.exception
    assert "in September" in at.title[0].value
    options = at.selectbox(key="subject").options
    first_subject = options[1]
    assert "maroon bells" in first_subject or "aspen" in first_subject or "lake" in first_subject
    season_buttons = [b for b in at.button if b.key and b.key.startswith("season-")]
    assert season_buttons, "expected in-season suggestions for September"

    season_buttons[0].click().run()
    assert not at.exception
    assert at.selectbox(key="subject").value in {"maroon bells", "aspen", "lake"}
    assert "September" in at.title[0].value

    at.select_slider(key="when").set_value("Jan").run()
    assert not at.exception
    assert any("No hexes match" in w.value for w in at.warning)  # maroon bells is never shot in January
