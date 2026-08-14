# Spatial Pipeline

```{image} https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg
:alt: Python Version
```
```{image} https://img.shields.io/badge/License-MIT-green.svg
:alt: License
```
```{image} https://img.shields.io/badge/GPU-RAPIDS%20cuGraph%20%7C%20cuML-76B900.svg
:alt: GPU RAPIDS
```
```{image} https://img.shields.io/badge/Out--of--Core-Dask%20%7C%20Zarr-orange.svg
:alt: Out-of-Core Dask/Zarr
```

**`spatial_pipeline`** is a high-performance computational engine for spatial transcriptomics (10x Genomics Xenium, Vizgen MERSCOPE, Nanostring CosMx, Stereo-seq). It provides ultra-fast spatial binning, GPU-accelerated spatial graph generation (RAPIDS cuGraph & cuML), Dask/Zarr out-of-core streaming for out-of-RAM datasets, Spatially Variable Gene (SVG) detection, and seamless AnnData / PyTorch Geometric interoperability.

---

## Key Highlights

- **⚡ Blazing Fast Spatial Binning**: Vectorized spatial hashing (Square, Hexagonal Axial, KD-Tree, Quadtree, and Adaptive Voronoi) processing millions of transcripts in seconds with minimal memory overhead.
- **🚀 GPU Graph Acceleration**: Build spatial radius and k-NN graphs and Graph Laplacians at scale on NVIDIA GPUs using RAPIDS cuGraph & cuML with seamless multi-core CPU fallbacks.
- **💾 Chunked Out-of-Core Processing**: Ingest and bin massive spatial coordinate datasets exceeding host RAM using **Dask Distributed DataFrames** and stream sub-matrices with **Zarr**.
- **🧬 Spatially Variable Gene (SVG) Discovery**: Vectorized computation of Moran's $I$, Geary's $C$, analytical $z$-scores, and Benjamini-Hochberg FDR $p$-values.
- **🔗 Ecosystem Interoperability**: Bi-directional conversions to [Scanpy](https://scanpy.readthedocs.io/) / [Squidpy](https://squidpy.readthedocs.io/) `AnnData` (.h5ad, .zarr) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) `Data` for Spatial Graph Neural Networks (GNNs).

---

## Quick Example

```python
import spatial_pipeline as sp

# 1. Ingest coordinate data with zero-copy PyArrow
df = sp.load_transcript_data("transcripts.csv", use_arrow=True)

# 2. Vectorized 10µm square spatial binning
sparse_counts, bin_coords = sp.compute_square_bins(df, resolution=10.0)

# 3. Build spatial adjacency graph (GPU or multi-core CPU)
adj_matrix, laplacian = sp.build_spatial_graph(bin_coords, max_distance=15.0, backend="auto")

# 4. Discover Spatially Variable Genes
svgs = sp.compute_spatially_variable_genes(
    sparse_counts, 
    gene_names=df["gene"].cat.categories, 
    adjacency_matrix=adj_matrix,
    top_n=10
)
print(svgs)

# 5. Export to AnnData (.h5ad) or chunked Zarr
adata = sp.to_anndata(sparse_counts, bin_coords, genes=df["gene"].cat.categories, adjacency_matrix=adj_matrix)
sp.to_zarr(adata, "spatial_processed.zarr")
```

---

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
tutorials/quickstart
```

```{toctree}
:maxdepth: 2
:caption: Advanced Guides

gpu_acceleration
out_of_core_dask_zarr
tutorials/out_of_core_streaming
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
