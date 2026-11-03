# Chunked Out-of-Core Processing (Dask / Zarr)

Large spatial transcriptomics datasets (such as 100M+ transcript observations) easily exceed available host RAM. `spatial_pipeline` provides out-of-core streaming and parallel matrix operations using **Dask Distributed DataFrames/Arrays** and **Zarr** hierarchical storage.

---

## 1. Dask Distributed Spatial Binning

For raw coordinate CSV files larger than RAM, `spatial_pipeline.dask_bin_transcripts()` partitions the file into chunks on disk and calculates spatial bin assignments and count aggregations out-of-core without exceeding memory limits:

```python
import spatial_pipeline as sp

# Out-of-core binning on massive transcript CSV
sparse_counts, bin_coords, unique_labels = sp.dask_bin_transcripts(
    "massive_transcripts.csv",
    resolution=10.0,
    min_counts=1,
    blocksize="128MB"
)
```

---

## 2. Zarr Chunked Storage & Streaming

`spatial_pipeline` allows saving AnnData objects directly to Zarr stores with custom chunking:

```python
# Convert and save to Zarr store
adata = sp.to_anndata(sparse_counts, bin_coords, labels=unique_labels)
sp.to_zarr(adata, "experiment.zarr", chunks=(10000, len(unique_labels)))
```

### Out-of-Core Batch Streaming

Stream slices of massive datasets for downstream deep learning models without loading full arrays into memory:

```python
for batch_X, batch_spatial, var_names in sp.stream_zarr_anndata("experiment.zarr", chunk_size=5000):
    # Process batch in GNN / neural network training loop
    train_step(batch_X, batch_spatial)
```

---

## 3. Parallel Dask Matrix Computations

Compute label dispersion, mean, variance, and sparsity across out-of-core Zarr datasets using Dask Array operations:

```python
metrics = sp.dask_compute_spatial_metrics("experiment.zarr", chunk_size=10000)

print("Mean label counts:", metrics["mean"])
print("Label variance:", metrics["var"])
print("Label sparsity:", metrics["sparsity"])
```
