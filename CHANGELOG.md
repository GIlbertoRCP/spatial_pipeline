# Changelog

All notable changes to the `spatial_pipeline` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-06

### Added
- **Spatially Variable Gene (SVG) Analysis**: Implemented vectorized `compute_morans_i()`, `compute_gearys_c()`, and `compute_spatially_variable_genes()` with FDR Benjamini-Hochberg adjusted p-values in `spatial_pipeline.graphs`.
- **AnnData SVG Integration**: Added `compute_svg` parameter to `to_anndata()` to populate `adata.var['morans_i']`, `adata.var['gearys_c']`, and `adata.var['morans_pvalue']`.
- **Expanded CLI Subcommands**: Added `spatial-pipeline svg` (spatial autocorrelation calculator), `spatial-pipeline export` (transcripts to `.h5ad`), and `spatial-pipeline ui` (Streamlit dashboard launcher).
- **Interactive SVG Dashboard Tab**: Integrated Spatially Variable Genes tab in `app.py` featuring Moran's I top 10 ranking bar chart and interactive autocorrelation data table.
- **Comprehensive Unit Test Suite**: Added `tests/test_graphs.py`, `tests/test_io.py`, `tests/test_out_of_core.py`, and `tests/test_cli.py` expanding test coverage across graph algorithms, I/O roundtrips, out-of-core tile halo stitching, and CLI subcommands.

### Changed
- Bumped project version to `0.3.0` in `pyproject.toml`.

---

## [0.2.0] - 2026-08-06

### Added
- **AnnData Ecosystem Interop (`.h5ad`)**: Added `to_anndata()`, `from_anndata()`, `save_anndata()`, and `read_anndata()` functions in `spatial_pipeline.io` following Scanpy and Squidpy conventions (`adata.X`, `adata.obsm['spatial']`, `adata.obsp['spatial_connectivities']`, `adata.obsp['spatial_laplacian']`).
- **Adaptive Voronoi Spatial Binning**: Added density-adaptive Voronoi region binning (`bin_voronoi()`) in `spatial_pipeline.binning`.
- **Count Thresholding**: Added `min_counts` parameter to `compute_square_bins()` to prune noisy low-density background bins.
- **PyTorch Geometric (PyG) GNN Export**: Added `to_pyg_data()` in `spatial_pipeline.graphs` to export graph expression, coordinates, and edge indices directly into `torch_geometric.data.Data`.
- **Graph Laplacian Spectral Analysis**: Added `compute_laplacian_spectrum()` to calculate top $K$ smallest eigenvalues of the Graph Laplacian for spatial connectivity profiling.
- **Benchmarking Scaling Suite**: Created `benchmarks/benchmark_scaling.py` script and scaling plot generator (`benchmarks/benchmark_scaling.png`).
- **Enhanced Streamlit Dashboard**: Added side-by-side comparison view (Raw Scatter vs. Binned Density Heatmap with dynamic resolution slider), live Graph Laplacian spectral eigenvalue plot, and direct AnnData `.h5ad` file download.

### Changed
- Bumped project version to `0.2.0` in `pyproject.toml`.
- Added `anndata>=0.9.0` and `h5py>=3.0.0` dependencies.

---

## [0.1.0] - 2026-08-06

### Added
- Out-of-Core Processing & `tracemalloc` Memory Profiling (`profile_out_of_core_execution`).
- Comparative Spatial Binning Benchmarking (`benchmark_binning_algorithms` comparing Spatial Hashing vs KD-Trees vs Quadtrees).
- Parallel Spatial Graph Construction (`build_spatial_graph` with multi-threading and multi-processing backends).
- Zero-copy PyArrow columnar memory data transfers (`load_transcript_arrow`, `arrow_table_to_numpy`).
- Modern Typer CLI interface (`spatial-pipeline run` & `spatial-pipeline benchmark`).
- Initial unit test suite and GitHub Actions CI/CD workflow (`.github/workflows/ci.yml`).
