# Large datasets: out-of-core processing

The spatial engine underneath the photo tools works on any labeled 2D points (`x`, `y`, `label`).
For inputs larger than memory it offers tiled processing with halo exchange and Dask-based binning.

The photo commands don't need these for a region like Colorado. They matter for the planned
full-scale run over all ~48M geotagged YFCC100M photos.

## Tiles with halo exchange

`partition_dataset_to_tiles` splits a point file into spatial tiles on disk. `process_tile_out_of_core`
bins one tile and builds its sub-graph using a halo of neighboring points, so bins on tile
borders get the same neighbors they would in a single pass. `stitch_out_of_core_results` merges
the tiles into global matrices.

```python
import os

import spatial_pipeline as sp

sp.partition_dataset_to_tiles("points.csv", tile_size=5000.0, partition_dir="tiles")

# Tiles are written as tiles/tile_<x>_<y>.csv
tiles = [tuple(int(v) for v in f[5:-4].split("_")) for f in os.listdir("tiles")
         if f.startswith("tile_") and f.endswith(".csv")]
sp.process_tiles_parallel_out_of_core(
    tiles, tile_size=5000.0, resolution=1000.0, max_dist=1050.0,
    partition_dir="tiles", tmp_out_dir="tile_results", n_jobs=4,
)
counts, centers, labels, adjacency, laplacian = sp.stitch_out_of_core_results("tile_results", resolution=1000.0)
```

`enforce_memory_limit` and `profile_out_of_core_execution` cap and measure peak memory, to check
that a run stays within a budget.

## Dask binning

`dask_bin_points` bins a CSV of `x, y, label` points with Dask DataFrames, a chunk at a time. It
needs the `scale` extra (`pip install -e .[scale]`):

```python
counts, centers, labels = sp.dask_bin_points("points.csv", resolution=1000.0, blocksize="128MB")
```
