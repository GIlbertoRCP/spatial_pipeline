"""
When to shoot: month-of-year and hour-of-day profiles per tag.

Counts are unique photographers per (tag, month) and (tag, hour), so one person's burst of
shots on a single outing counts once. Each profile is compared with the profile of all
photos: `lift` > 1 means a tag is photographed unusually often in that month or hour,
which separates true seasonality (elk rut in September) from the general summer peak in
people being outdoors.

Times are local wall-clock times as stored in `taken_at`. A time of exactly 00:00:00 is
treated as "date only" (iNaturalist observations without a time, cameras without a clock)
and excluded from hour profiles, though the date still counts toward month profiles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .photos import DEFAULT_STOP_TAGS, explode_tags

ALL_LABEL = "(all photos)"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TIMING_COLUMNS = ["label", "kind", "bin", "photographers", "share", "lift"]

_BINS = {"month": np.arange(1, 13), "hour": np.arange(24)}


def _local_times(taken_at: pd.Series) -> pd.Series:
    taken = pd.to_datetime(taken_at, errors="coerce")
    if isinstance(taken.dtype, pd.DatetimeTZDtype):
        taken = taken.dt.tz_localize(None)  # keep the wall-clock time
    return taken


def _profile(keys: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Unique photographers per (label, bin), with each label's share across bins."""
    counts = keys.drop_duplicates().groupby(["label", "bin"]).size().rename("photographers")
    full = pd.MultiIndex.from_product([counts.index.unique("label"), _BINS[kind]], names=["label", "bin"])
    counts = counts.reindex(full, fill_value=0)
    totals = counts.groupby(level="label").transform("sum")
    out = counts.to_frame().assign(share=(counts / totals.where(totals > 0)).fillna(0.0)).reset_index()
    out["kind"] = kind
    return out


def compute_timing(
    photos: pd.DataFrame,
    labels: list[str] | np.ndarray | None = None,
    stop_tags: frozenset[str] | set[str] = DEFAULT_STOP_TAGS,
) -> pd.DataFrame:
    """
    Month-of-year and hour-of-day photographer profiles for each tag and for all photos.

    Args:
        photos: Photo table with 'user_id', 'taken_at' and 'tags' columns.
        labels: Tags to profile (default: every tag that survives `explode_tags`).
        stop_tags: Tags to ignore.

    Returns:
        pd.DataFrame: Long table with columns ['label', 'kind', 'bin', 'photographers', 'share',
        'lift'], where kind is 'month' (bin 1-12) or 'hour' (bin 0-23). The ALL_LABEL rows hold
        the baseline profile of all photos; `lift` is a tag's share divided by the baseline share.
    """
    photos = photos.reset_index(drop=True)
    taken = _local_times(photos["taken_at"])
    users = pd.Series(pd.factorize(photos["user_id"])[0], index=photos.index)
    has_date = taken.notna()
    has_time = has_date & ~((taken.dt.hour == 0) & (taken.dt.minute == 0) & (taken.dt.second == 0))
    bins = {"month": taken.dt.month, "hour": taken.dt.hour}
    valid = {"month": has_date, "hour": has_time}

    tags = explode_tags(photos, stop_tags=stop_tags)
    if labels is not None:
        tags = tags[tags["label"].isin(set(labels))]
    idx = tags["photo_idx"].to_numpy()

    frames = []
    for kind in ("month", "hour"):
        ok = valid[kind].to_numpy()
        base = pd.DataFrame({"label": ALL_LABEL, "user": users[ok], "bin": bins[kind][ok].astype(int)})
        tag_ok = ok[idx]
        per_tag = pd.DataFrame({
            "label": tags["label"].to_numpy()[tag_ok],
            "user": users.to_numpy()[idx][tag_ok],
            "bin": bins[kind].to_numpy()[idx][tag_ok].astype(int),
        })
        if base.empty:
            continue
        profile = _profile(pd.concat([base, per_tag], ignore_index=True), kind)
        baseline = profile[profile["label"] == ALL_LABEL].set_index("bin")["share"]
        base_share = profile["bin"].map(baseline).to_numpy()
        profile["lift"] = np.divide(
            profile["share"].to_numpy(), base_share, out=np.zeros(len(profile)), where=base_share > 0
        )
        frames.append(profile)

    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(TIMING_COLUMNS, ["object", "object", "int64", "int64", "float64", "float64"])})
    return pd.concat(frames, ignore_index=True)[TIMING_COLUMNS]


def peak_bins(timing: pd.DataFrame, label: str, kind: str = "month", n: int = 3) -> list[int]:
    """
    The `n` bins with the most photographers for a tag (ties broken by lift), most first.

    Args:
        timing: Output of `compute_timing`.
        label: Tag to summarize.
        kind: 'month' or 'hour'.
        n: Number of bins to return.

    Returns:
        list[int]: Bin values (months 1-12 or hours 0-23); empty if the tag has no data.
    """
    rows = timing[(timing["label"] == label) & (timing["kind"] == kind) & (timing["photographers"] > 0)]
    rows = rows.sort_values(["photographers", "lift"], ascending=False)
    return [int(b) for b in rows["bin"].head(n)]


def describe_peaks(timing: pd.DataFrame, label: str, n: int = 3) -> str:
    """
    Human-readable peak months and hours, e.g. 'Sep, Oct, Aug; 7:00, 18:00, 8:00'.

    Args:
        timing: Output of `compute_timing`.
        label: Tag to summarize.
        n: Number of months and hours to list.

    Returns:
        str: Peak summary ('' when the tag has no timing data).
    """
    months = ", ".join(MONTH_NAMES[m - 1] for m in peak_bins(timing, label, "month", n))
    hours = ", ".join(f"{h}:00" for h in peak_bins(timing, label, "hour", n))
    return "; ".join(part for part in (months, hours) if part)


def save_timing(timing: pd.DataFrame, out_dir: str) -> None:
    """
    Writes timing profiles to `<out_dir>/timing.parquet`.

    Args:
        timing: Output of `compute_timing`.
        out_dir: Target directory (created if needed).
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    timing.to_parquet(path / "timing.parquet", index=False)
