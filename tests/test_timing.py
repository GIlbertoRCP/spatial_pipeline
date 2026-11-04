from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spatial_pipeline.hotspots import compute_hotspots, load_hotspots, save_hotspots
from spatial_pipeline.timing import (
    ALL_LABEL,
    compute_timing,
    describe_peaks,
    peak_bins,
    save_timing,
)


def make_photos(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0

    def add(user: str, when: str, tags: str) -> None:
        nonlocal pid
        rows.append({"photo_id": str(pid), "user_id": user, "lon": -105.58, "lat": 40.38,
                     "taken_at": pd.Timestamp(when), "tags": tags})
        pid += 1

    # Baseline: people photograph all year, more in summer, mostly midday.
    for u in range(400):
        month = int(rng.choice(np.arange(1, 13), p=np.array([1, 1, 2, 3, 5, 8, 9, 8, 5, 3, 2, 1]) / 48))
        add(f"b{u}", f"2021-{month:02d}-15 {int(rng.integers(10, 16)):02d}:30:00", "plants")
    # Elk: September/October around dawn and dusk.
    for u in range(60):
        month = int(rng.choice([9, 9, 9, 10, 10]))
        hour = int(rng.choice([6, 7, 7, 18]))
        add(f"e{u}", f"2021-{month:02d}-20 {hour:02d}:10:00", "elk,mammals")
    # One enthusiast with 200 elk photos on a single outing: still one photographer.
    for _ in range(200):
        add("burst", "2021-03-02 12:00:00", "elk,mammals")
    # Aspen in fall color, some with date only (no time).
    for u in range(40):
        when = "2021-09-25 00:00:00" if u % 4 == 0 else "2021-09-25 11:45:00"
        add(f"a{u}", when, "quaking aspen,plants")
    return pd.DataFrame(rows)


def lookup(timing: pd.DataFrame, label: str, kind: str, b: int) -> pd.Series:
    return timing[(timing["label"] == label) & (timing["kind"] == kind) & (timing["bin"] == b)].iloc[0]


def test_profiles_have_full_bins_and_shares_sum_to_one():
    timing = compute_timing(make_photos())
    for label in (ALL_LABEL, "elk", "quaking aspen"):
        months = timing[(timing["label"] == label) & (timing["kind"] == "month")]
        hours = timing[(timing["label"] == label) & (timing["kind"] == "hour")]
        assert sorted(months["bin"]) == list(range(1, 13))
        assert sorted(hours["bin"]) == list(range(24))
        assert months["share"].sum() == pytest.approx(1.0)
        assert hours["share"].sum() == pytest.approx(1.0)


def test_peaks_and_lift_capture_seasonality():
    timing = compute_timing(make_photos())
    assert peak_bins(timing, "elk", "month", 2) == [9, 10]
    assert peak_bins(timing, "elk", "hour", 1) == [7]
    assert peak_bins(timing, "quaking aspen", "month", 1) == [9]
    assert lookup(timing, "elk", "month", 9)["lift"] > 2  # baseline includes the elk photos themselves
    assert lookup(timing, "elk", "month", 7)["lift"] == 0
    assert describe_peaks(timing, "elk", 2) == "Sep, Oct; 7:00, 6:00"
    assert describe_peaks(timing, "no such tag") == ""


def test_photographers_counted_once_per_bin_and_date_only_excluded_from_hours():
    timing = compute_timing(make_photos())
    assert lookup(timing, "elk", "month", 3)["photographers"] == 1  # the 200-photo burst
    aspen_months = timing[(timing["label"] == "quaking aspen") & (timing["kind"] == "month")]["photographers"].sum()
    aspen_hours = timing[(timing["label"] == "quaking aspen") & (timing["kind"] == "hour")]["photographers"].sum()
    assert aspen_months == 40
    assert aspen_hours == 30  # the 10 date-only photos have no usable hour
    assert lookup(timing, "quaking aspen", "hour", 0)["photographers"] == 0


def test_labels_filter_tz_aware_times_and_empty_input():
    photos = make_photos()
    timing = compute_timing(photos, labels=["elk"])
    assert set(timing["label"]) == {ALL_LABEL, "elk"}

    aware = photos.assign(taken_at=photos["taken_at"].dt.tz_localize("America/Denver"))
    assert peak_bins(compute_timing(aware), "elk", "hour", 1) == [7]  # wall-clock hour kept

    empty = compute_timing(photos.assign(taken_at=pd.NaT))
    assert empty.empty and list(empty.columns) == ["label", "kind", "bin", "photographers", "share", "lift"]


def test_timing_roundtrip_with_hotspots(tmp_path):
    photos = make_photos()
    out = str(tmp_path / "hot")
    save_hotspots(compute_hotspots(photos, min_label_photographers=1, min_hex_photographers=1), out)
    assert load_hotspots(out)["timing"].empty  # older outputs without timing still load

    save_timing(compute_timing(photos), out)
    loaded = load_hotspots(out)["timing"]
    assert os.path.exists(os.path.join(out, "timing.parquet"))
    assert peak_bins(loaded, "elk", "month", 1) == [9]
