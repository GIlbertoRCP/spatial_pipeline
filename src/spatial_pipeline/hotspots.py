"""
Photo hotspot analysis: hex binning, photographer-weighted counts, and spatial tag ranking.

Counts are unique photographers, not photos: one person uploading 500 shots from their
backyard is one vote for that hex, not 500. Tags are ranked by spatial autocorrelation
(Moran's I) of their per-hex share, which surfaces tags that concentrate in specific
places ('waterfall', 'redrocks') rather than tags that simply follow where people are.
Local Getis-Ord Gi* then marks *which* hexes are statistically significant hot spots for
each tag and for overall photographer density.

All statistics compare active (photographed) hexes with each other: unphotographed land
is not part of the graph, so "hot" means hot relative to other places people photograph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from .binning import bin_hexagonal, get_bin_centers
from .geo import DEFAULT_CRS, project_lonlat, unproject_xy
from .graphs import (
    build_spatial_graph,
    compute_getis_ord_gi_star,
    compute_spatially_variable_labels,
    fdr_bh,
)
from .photos import DEFAULT_STOP_TAGS, explode_tags
from .timing import ALL_LABEL


@dataclass
class HotspotResult:
    """
    Output of `compute_hotspots`.

    Attributes:
        hexes: One row per active hex: hex_id, q, r, x, y, lon, lat, photos, photographers,
            plus density_gi_z / density_fdr (Gi* of log photographer counts; NaN for hexes
            outside the statistics).
        label_counts: Long table of (hex_id, label, photographers) for the selected labels.
        local_hotspots: Significant hot spots per tag: (hex_id, label, gi_z, fdr_p_value),
            with gi_z > 0 and FDR-adjusted p-value below `alpha` (corrected per tag).
        ranking: Labels ranked by Moran's I with Geary's C, z-score, FDR p-value,
            total photographers, number of hexes, and number of significant hot-spot hexes.
        matrix: Sparse (num_hexes, num_labels) matrix of unique-photographer counts.
        labels: Label names, aligned with matrix columns.
        adjacency: Hex adjacency (immediate neighbors) over the hexes used for statistics.
        stats_hex_ids: hex_id of each row/column in `adjacency`.
        resolution: Hex center-to-center distance in CRS units (meters by default).
        crs: Projected CRS used for binning.
        photo_hex_ids: hex_id of each input photo (aligned with the photo table's row order).
        num_photographers: Distinct photographers across all input photos.
    """

    hexes: pd.DataFrame
    label_counts: pd.DataFrame
    local_hotspots: pd.DataFrame
    ranking: pd.DataFrame
    matrix: csr_matrix
    labels: np.ndarray
    adjacency: csr_matrix
    stats_hex_ids: np.ndarray
    resolution: float
    crs: str
    photo_hex_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    num_photographers: int = 0


def _unique_pair_counts(group_codes: np.ndarray, user_codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Counts distinct users per group code."""
    if len(group_codes) == 0:
        return np.zeros(n_groups, dtype=np.int64)
    n_users = int(user_codes.max()) + 1
    pairs = np.unique(group_codes.astype(np.int64) * n_users + user_codes.astype(np.int64))
    return np.bincount(pairs // n_users, minlength=n_groups)


def compute_hotspots(
    photos: pd.DataFrame,
    resolution: float = 1000.0,
    crs: str = DEFAULT_CRS,
    max_labels: int = 300,
    min_label_photographers: int = 10,
    min_hex_photographers: int = 3,
    neighbor_rings: int = 1,
    normalize: Literal["share", "count"] = "share",
    alpha: float = 0.05,
    stop_tags: frozenset[str] | set[str] = DEFAULT_STOP_TAGS,
) -> HotspotResult:
    """
    Bins photos into hexes and ranks tags by how spatially clustered they are.

    Args:
        photos: Photo table with 'user_id', 'lon', 'lat', 'tags' columns.
        resolution: Hex center-to-center distance in CRS units (default 1000 m).
        crs: Projected CRS for binning (default EPSG:5070, CONUS Albers).
        max_labels: Keep at most this many tags (most photographers first).
        min_label_photographers: Minimum distinct photographers for a tag to be analyzed.
        min_hex_photographers: Hexes with fewer distinct photographers are excluded from the
            statistics (their tag shares are too noisy) but still appear in `hexes`.
        neighbor_rings: Hexes within this many rings of each other are neighbors (1 = touching).
            Hexes with no neighbors are excluded from the statistics: with binary weights,
            isolated hexes inflate Moran's I (N / S0 grows while no neighbor pairs are added).
        normalize: 'share' ranks tags by their fraction of each hex's photographers;
            'count' uses raw photographer counts (dominated by overall photo density).
        alpha: FDR level for local Gi* hot spots.
        stop_tags: Tags to ignore.

    Returns:
        HotspotResult: Hex table, per-hex tag counts, tag ranking, and the underlying matrices.

    Raises:
        ValueError: If the photo table is empty.
    """
    if len(photos) == 0:
        raise ValueError("No photos to analyze.")

    photos = photos.reset_index(drop=True)
    x, y = project_lonlat(photos["lon"].to_numpy(), photos["lat"].to_numpy(), crs)
    qr = bin_hexagonal(x, y, resolution)
    unique_qr, hex_of_photo = np.unique(qr, axis=0, return_inverse=True)
    hex_of_photo = hex_of_photo.ravel()
    n_hex = len(unique_qr)

    user_codes, _ = pd.factorize(photos["user_id"])
    photos_per_hex = np.bincount(hex_of_photo, minlength=n_hex)
    photographers_per_hex = _unique_pair_counts(hex_of_photo, user_codes, n_hex)

    centers = get_bin_centers(unique_qr, "hex", resolution)
    lon, lat = unproject_xy(centers[:, 0], centers[:, 1], crs)
    hexes = pd.DataFrame({
        "hex_id": np.arange(n_hex),
        "q": unique_qr[:, 0],
        "r": unique_qr[:, 1],
        "x": centers[:, 0],
        "y": centers[:, 1],
        "lon": lon,
        "lat": lat,
        "photos": photos_per_hex,
        "photographers": photographers_per_hex,
    })

    # (hex, label, user) triples: each photographer counts once per tag per hex.
    tags = explode_tags(photos, stop_tags=stop_tags)
    idx = tags["photo_idx"].to_numpy()
    triples = pd.DataFrame({
        "hex_id": hex_of_photo[idx],
        "label": tags["label"].to_numpy(),
        "user": user_codes[idx],
    }).drop_duplicates()

    label_users = triples.drop_duplicates(["label", "user"])["label"].value_counts()
    selected = label_users[label_users >= min_label_photographers].head(max_labels)
    labels = np.asarray(selected.index, dtype=object)

    triples = triples[triples["label"].isin(selected.index)]
    label_codes = pd.Categorical(triples["label"], categories=labels).codes
    matrix = csr_matrix(
        (np.ones(len(triples), dtype=np.int32), (triples["hex_id"].to_numpy(), label_codes)),
        shape=(n_hex, len(labels)),
        dtype=np.int32,
    )

    label_counts = (
        triples.groupby(["hex_id", "label"], observed=True).size().rename("photographers").reset_index()
    )

    # Ring k of a hex grid lies within k * resolution of the center hex.
    stats_hex_ids = np.flatnonzero(photographers_per_hex >= min_hex_photographers)
    adjacency, _ = build_spatial_graph(
        centers[stats_hex_ids], max_distance=resolution * (neighbor_rings + 0.05), backend="cpu"
    )
    connected = np.asarray(adjacency.sum(axis=1)).ravel() > 0
    stats_hex_ids = stats_hex_ids[connected]
    adjacency = csr_matrix(adjacency[connected][:, connected])

    stats_matrix = matrix[stats_hex_ids]
    if normalize == "share":
        stats_matrix = diags(1.0 / photographers_per_hex[stats_hex_ids]) @ stats_matrix

    if len(labels) and len(stats_hex_ids) > 1:
        ranking = compute_spatially_variable_labels(stats_matrix, labels, adjacency)
    else:
        ranking = pd.DataFrame(columns=["label", "morans_i", "gearys_c", "z_score", "p_value", "fdr_p_value"])
    ranking["photographers"] = ranking["label"].map(selected).astype("int64")
    ranking["hexes"] = ranking["label"].map(label_counts["label"].value_counts()).fillna(0).astype("int64")

    hexes["density_gi_z"] = np.nan
    hexes["density_fdr"] = np.nan
    local_hotspots = pd.DataFrame({
        "hex_id": pd.Series(dtype="int64"), "label": pd.Series(dtype="object"),
        "gi_z": pd.Series(dtype="float64"), "fdr_p_value": pd.Series(dtype="float64"),
    })
    if len(stats_hex_ids) > 1:
        density_z, density_p = compute_getis_ord_gi_star(np.log1p(photographers_per_hex[stats_hex_ids]), adjacency)
        hexes.loc[stats_hex_ids, "density_gi_z"] = density_z
        hexes.loc[stats_hex_ids, "density_fdr"] = fdr_bh(density_p)

        if len(labels):
            gi_z, gi_p = compute_getis_ord_gi_star(stats_matrix, adjacency)
            gi_fdr = fdr_bh(gi_p, axis=0)  # one family of tests per tag
            rows, cols = np.nonzero((gi_z > 0) & (gi_fdr < alpha))
            local_hotspots = pd.DataFrame({
                "hex_id": stats_hex_ids[rows],
                "label": labels[cols],
                "gi_z": gi_z[rows, cols],
                "fdr_p_value": gi_fdr[rows, cols],
            })
    ranking["hotspot_hexes"] = (
        ranking["label"].map(local_hotspots["label"].value_counts()).fillna(0).astype("int64")
    )

    return HotspotResult(
        hexes=hexes,
        label_counts=label_counts,
        local_hotspots=local_hotspots,
        ranking=ranking,
        matrix=matrix,
        labels=labels,
        adjacency=adjacency,
        stats_hex_ids=stats_hex_ids,
        resolution=float(resolution),
        crs=crs,
        photo_hex_ids=hex_of_photo,
        num_photographers=int(user_codes.max()) + 1 if len(user_codes) else 0,
    )


def hex_polygons_lonlat(x: np.ndarray, y: np.ndarray, resolution: float, crs: str = DEFAULT_CRS) -> list[list[list[float]]]:
    """
    Builds lon/lat outlines of pointy-topped hexes (matching `bin_hexagonal`) for map rendering.

    Args:
        x: 1D array of hex center x coordinates in `crs`.
        y: 1D array of hex center y coordinates in `crs`.
        resolution: Hex center-to-center distance.
        crs: Projected CRS of the centers.

    Returns:
        list[list[list[float]]]: One ring of six [lon, lat] vertices per hex.
    """
    radius = resolution / np.sqrt(3.0)
    angles = np.deg2rad(30.0 + 60.0 * np.arange(6))
    vx = np.asarray(x)[:, None] + radius * np.cos(angles)[None, :]
    vy = np.asarray(y)[:, None] + radius * np.sin(angles)[None, :]
    lon, lat = unproject_xy(vx.ravel(), vy.ravel(), crs)
    ring = np.stack([lon, lat], axis=1).reshape(len(vx), 6, 2)
    return ring.tolist()


PHOTO_SAMPLE_COLUMNS = [
    "hex_id", "label", "photo_id", "thumbnail_url", "page_url", "owner_name", "attribution",
    "title", "license", "taken_at",
]


def sample_photos(
    photos: pd.DataFrame,
    result: HotspotResult,
    per_group: int = 6,
    stop_tags: frozenset[str] | set[str] = DEFAULT_STOP_TAGS,
) -> pd.DataFrame:
    """
    Picks up to `per_group` photos with thumbnails for every (hex, tag) and every hex overall,
    each from a different photographer, for display in the map.

    Args:
        photos: The photo table passed to `compute_hotspots` (same row order).
        result: Output of `compute_hotspots`.
        per_group: Photos to keep per (hex, tag) and per hex.
        stop_tags: Tags to ignore (should match `compute_hotspots`).

    Returns:
        pd.DataFrame: Columns PHOTO_SAMPLE_COLUMNS; label is ALL_LABEL for the per-hex sample.
        Empty if the photo table has no thumbnail URLs (e.g. older YFCC tables).
    """
    photos = photos.reset_index(drop=True)
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in PHOTO_SAMPLE_COLUMNS})
    if "thumbnail_url" not in photos or len(result.photo_hex_ids) != len(photos):
        return empty

    has_thumb = photos["thumbnail_url"].fillna("").astype(str).str.startswith("http").to_numpy()
    if not has_thumb.any():
        return empty

    def col(name: str) -> pd.Series:
        return photos[name] if name in photos else pd.Series("", index=photos.index)

    base = pd.DataFrame({
        "hex_id": result.photo_hex_ids,
        "photo_id": photos["photo_id"].astype(str),
        "user": photos["user_id"].astype(str),
        "thumbnail_url": col("thumbnail_url").fillna("").astype(str),
        "page_url": col("page_url").fillna("").astype(str),
        "owner_name": col("owner_name").fillna("").astype(str),
        "attribution": col("attribution").fillna("").astype(str),
        "title": col("title").fillna("").astype(str),
        "license": col("license").fillna("").astype(str),
        "taken_at": pd.to_datetime(photos["taken_at"], errors="coerce").astype(str).replace("NaT", ""),
    })[has_thumb]

    tags = explode_tags(photos, stop_tags=stop_tags)
    tags = tags[tags["label"].isin(set(result.labels)) & has_thumb[tags["photo_idx"].to_numpy()]]
    per_tag = base.loc[tags["photo_idx"].to_numpy()].assign(label=tags["label"].to_numpy())
    overall = base.assign(label=ALL_LABEL)

    samples = pd.concat([overall, per_tag], ignore_index=True)
    samples = samples.drop_duplicates(["hex_id", "label", "user"])
    samples = samples.groupby(["hex_id", "label"], sort=False).head(per_group)
    return samples[PHOTO_SAMPLE_COLUMNS].reset_index(drop=True)


def save_photo_samples(samples: pd.DataFrame, out_dir: str) -> None:
    """
    Writes photo samples to `<out_dir>/photo_samples.parquet`.

    Args:
        samples: Output of `sample_photos`.
        out_dir: Target directory (created if needed).
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(path / "photo_samples.parquet", index=False)


def save_hotspots(result: HotspotResult, out_dir: str, extra_meta: dict[str, object] | None = None) -> None:
    """
    Writes hotspot outputs to a directory: hexes.parquet, label_counts.parquet,
    local_hotspots.parquet, ranking.csv, meta.json.

    Args:
        result: Output of `compute_hotspots`.
        out_dir: Target directory (created if needed).
        extra_meta: Extra fields to record in meta.json (e.g. input path, bbox).
    """
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    result.hexes.to_parquet(path / "hexes.parquet", index=False)
    result.label_counts.to_parquet(path / "label_counts.parquet", index=False)
    result.local_hotspots.to_parquet(path / "local_hotspots.parquet", index=False)
    result.ranking.to_csv(path / "ranking.csv", index=False)
    meta: dict[str, object] = {
        "resolution": result.resolution,
        "crs": result.crs,
        "num_hexes": len(result.hexes),
        "num_stats_hexes": len(result.stats_hex_ids),
        "num_labels": len(result.labels),
        "num_photos": int(result.hexes["photos"].sum()),
        "num_photographers": result.num_photographers,
        "num_hotspot_hexes": int(result.local_hotspots["hex_id"].nunique()),
    }
    if extra_meta:
        meta.update(extra_meta)
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_hotspots(out_dir: str) -> dict[str, object]:
    """
    Loads the files written by `save_hotspots`.

    Args:
        out_dir: Directory containing the hotspot outputs.

    Returns:
        dict[str, object]: Keys 'hexes', 'label_counts', 'local_hotspots', 'ranking' (DataFrames)
            and 'meta' (dict), plus 'timing' and 'photo_samples' (DataFrames). Outputs written
            before those files existed load them as empty tables.
    """
    path = Path(out_dir)
    local_path = path / "local_hotspots.parquet"
    local = (
        pd.read_parquet(local_path) if local_path.exists()
        else pd.DataFrame(columns=["hex_id", "label", "gi_z", "fdr_p_value"])
    )
    return {
        "hexes": pd.read_parquet(path / "hexes.parquet"),
        "label_counts": pd.read_parquet(path / "label_counts.parquet"),
        "local_hotspots": local,
        "timing": (
            pd.read_parquet(path / "timing.parquet") if (path / "timing.parquet").exists()
            else pd.DataFrame(columns=["label", "kind", "bin", "photographers", "share", "lift"])
        ),
        "photo_samples": (
            pd.read_parquet(path / "photo_samples.parquet") if (path / "photo_samples.parquet").exists()
            else pd.DataFrame(columns=PHOTO_SAMPLE_COLUMNS)
        ),
        "ranking": pd.read_csv(path / "ranking.csv"),
        "meta": json.loads((path / "meta.json").read_text(encoding="utf-8")),
    }
