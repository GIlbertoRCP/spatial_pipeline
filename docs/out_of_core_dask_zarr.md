# Large datasets: out-of-core processing

The spatial engine underneath the photo tools works on any labeled 2D points (`x`, `y`, `label`).
For inputs larger than memory it offers tiled processing with halo exchange, Dask-based binning
and Zarr streaming.

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

`run_pipeline.py` wraps these steps into one command.

`enforce_memory_limit` and `profile_out_of_core_execution` cap and measure peak memory, to check
that a run stays within a budget.

## Dask binning

`dask_bin_transcripts` bins a CSV of `x, y, label` points with Dask DataFrames, a chunk at a time:

```python
counts, centers, labels = sp.dask_bin_transcripts("points.csv", resolution=1000.0, blocksize="128MB")
```

## Zarr storage and streaming

Binned matrices can be stored as chunked Zarr (via AnnData) and streamed back in batches, for
example to train a graph neural network without loading everything into memory:

```python
adata = sp.to_anndata(counts, centers, labels=labels, adjacency_matrix=adjacency)
sp.to_zarr(adata, "hexes.zarr", chunks=(10000, len(labels)))

for batch_X, batch_coords, names in sp.stream_zarr_anndata("hexes.zarr", chunk_size=5000):
    ...

metrics = sp.dask_compute_spatial_metrics("hexes.zarr", chunk_size=10000)  # mean, var, sparsity
```

Streaming needs a dense `X`. Sparse stores raise a clear error.
