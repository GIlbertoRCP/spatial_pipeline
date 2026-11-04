import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from spatial_pipeline.out_of_core import (
    HAS_DASK,
    dask_bin_points,
    load_tile_with_halo,
    partition_dataset_to_tiles,
    process_tile_out_of_core,
    profile_memory,
    profile_out_of_core_execution,
    stitch_out_of_core_results,
)

from tests.synthetic import write_synthetic_points


def test_profile_memory_context_manager():
    with profile_memory() as metrics:
        arr = np.zeros((1000, 1000), dtype=np.float64)
        _ = arr.sum()

    assert "peak_mb" in metrics
    assert "elapsed_sec" in metrics
    assert metrics["peak_mb"] > 0.0
    assert metrics["elapsed_sec"] >= 0.0


def test_out_of_core_partitioning_and_stitching():
    tmp_dir = tempfile.mkdtemp()
    try:
        mock_csv = os.path.join(tmp_dir, "points.csv")
        partition_dir = os.path.join(tmp_dir, "tiles")

        write_synthetic_points(mock_csv, num_points=1000)

        # Partition into 100-unit tiles
        partition_dataset_to_tiles(mock_csv, tile_size=100.0, partition_dir=partition_dir)

        assert os.path.exists(partition_dir)
        assert os.path.exists(os.path.join(partition_dir, "labels.npy"))
        assert os.path.exists(os.path.join(partition_dir, "offset.npy"))

        # Process a tile out of core
        tile_files = [f for f in os.listdir(partition_dir) if f.startswith("tile_") and f.endswith(".csv")]
        assert len(tile_files) > 0

        tile_parts = tile_files[0].replace("tile_", "").replace(".csv", "").split("_")
        tx, ty = int(tile_parts[0]), int(tile_parts[1])

        tmp_out_dir = os.path.join(tmp_dir, "out_tiles")
        os.makedirs(tmp_out_dir, exist_ok=True)

        process_tile_out_of_core(
            tile_x=tx,
            tile_y=ty,
            tile_size=100.0,
            resolution=10.0,
            max_dist=15.0,
            partition_dir=partition_dir,
            tmp_out_dir=tmp_out_dir
        )
        tile_out_files = [f for f in os.listdir(tmp_out_dir) if f.startswith("counts_")]
        assert len(tile_out_files) > 0

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_profile_out_of_core_execution():
    tmp_dir = tempfile.mkdtemp()
    try:
        mock_csv = os.path.join(tmp_dir, "bench_points.csv")
        write_synthetic_points(mock_csv, num_points=800)

        report = profile_out_of_core_execution(
            file_path=mock_csv,
            tile_size=150.0,
            resolution=15.0,
            max_dist=20.0
        )

        assert "partitioning" in report
        assert "tile_processing" in report
        assert "stitching" in report
        assert "summary" in report
        assert report["summary"]["total_elapsed_sec"] > 0.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_dask_bin_points():
    tmp_dir = tempfile.mkdtemp()
    try:
        mock_csv = os.path.join(tmp_dir, "dask_points.csv")
        write_synthetic_points(mock_csv, num_points=1200)

        sparse_counts, bin_centers, unique_labels = dask_bin_points(
            mock_csv,
            resolution=25.0,
            min_counts=1
        )

        assert sparse_counts.shape[0] > 0
        assert sparse_counts.shape[1] == len(unique_labels)
        assert bin_centers.shape[0] == sparse_counts.shape[0]
        assert bin_centers.shape[1] == 2
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

