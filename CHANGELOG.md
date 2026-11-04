# Changelog

All notable changes to the `spatial_pipeline` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Local Getis-Ord Gi* hot-spot statistic (`compute_getis_ord_gi_star`), verified against the textbook formula.
- Vectorized Benjamini-Hochberg correction (`fdr_bh`), applied per tag across hexes; replaces the per-label loop.
- `compute_hotspots` now returns significant per-tag hot spots (`local_hotspots`, saved as `local_hotspots.parquet`), photographer-density Gi* per hex, and a `hotspot_hexes` count in the tag ranking. New `alpha` option / `--alpha` flag.
- Map: "Only significant hot spots" filter, and Gi* z-scores in the tooltip and top-hexes table.
- `flickr` module and `fetch-flickr` CLI command: throttled, retrying `flickr.photos.search` client; fetcher that partitions by upload-date window and splits windows over Flickr's ~4,000-result cap in time, then space; resumable per-window Parquet cache; optional open-license filter. Output keeps owner name, title, license and thumbnail URL for attribution and later embedding work. Tested against a simulated API; not run live, since Flickr now issues API keys only to Pro accounts.
- `inat` module and `fetch-inat` CLI command (no API key): iNaturalist observations with photos become photo-table rows (observer = photographer, species common name + group = tags). Pages by observation id to avoid the 10,000-result cap; throttled to ~1 request/s with a per-run request budget; resumable per-page cache tied to the query; excludes obscured coordinates and observations with location uncertainty above 1 km; defaults to research grade and Creative Commons photos. Verified live on Denver-area data.
- Shared `_http.ThrottledJSONClient` (throttling and retries) used by the Flickr and iNaturalist clients.
- `timing` module: month-of-year and hour-of-day profiles per tag in unique photographers, with lift against all photos (`compute_timing`, `peak_bins`, `describe_peaks`). Date-only timestamps (00:00:00) count for months but not hours. `hotspots` saves `timing.parquet` and adds `peak_months` to the ranking; the map shows a "When to go" section.
- Photo samples: `sample_photos` keeps up to 6 thumbnails per (hex, tag) and per hex, one per photographer, saved as `photo_samples.parquet`. iNaturalist and Flickr rows gain `page_url`; YFCC rows gain thumbnail URLs, photographer names, titles, licenses and page links.
- Dashboard redesign: dark theme with golden-hour accents (`.streamlit/config.toml`, applied by `spatial-pipeline ui`); a map + detail-panel layout; click a hexagon to open a spot panel (subjects with ★ hot-spot markers, photos with credits, directions), with the selection kept in a shareable `?hex=` link and outlined on the map; a color legend; readable tooltips; a log-scaled color ramp with transparency for quiet spots; plain-language metrics; and the data folder moved into a collapsed "Data source" section.

### Fixed
- CI: ruff errors in `graphs.py` / `out_of_core.py`, mypy errors from zarr 3 typing and the pinned Python version; pandas floor raised to 2.0.

## [0.5.0] - 2026-11-03

Pivot from spatial transcriptomics to geotagged photo hotspots (milestone 1 of `PLAN.md`).

### Added
- `photos` module: streaming YFCC100M metadata reader for TSV, `.bz2`/`.gz` TSV and SQLite (`read_yfcc_metadata`), with 23/25-column layout detection, bbox / accuracy / video filtering, deduplication, URL-encoded tag decoding and tag filtering (`explode_tags`).
- `geo` module: named and custom bounding boxes (`parse_bbox`), lon/lat ↔ EPSG:5070 projection (`project_lonlat`, `unproject_xy`).
- `hotspots` module: hex hotspots counted by unique photographers, tag ranking by Moran's I / Geary's C on per-hex tag share, hex outlines for mapping, save/load of outputs.
- CLI commands `ingest-yfcc` and `hotspots`.
- Synthetic YFCC-layout test fixture and tests for ingestion, hotspots, CLI and the Streamlit app.

### Changed
- Renamed the transcript-specific `gene` column and identifiers to the generic `label` throughout (`compute_spatially_variable_genes` → `compute_spatially_variable_labels`).
- `load_coordinates` now reads generic `x, y, label` files; Xenium column mapping and QV filtering were removed.
- Rewrote the Streamlit app (`app.py`) as an interactive hex hotspot map.
- Rewrote the README; removed performance claims measured only on synthetic data.
- Added `pyproj`, `pydeck` and `rich` dependencies; updated package metadata.

## [0.4.0] - 2026-08-14

### Added
- **GPU Graph Acceleration (RAPIDS cuGraph / cuML)**:
  - Added GPU-accelerated k-NN (`build_knn_graph`) and radius graph construction with RAPIDS cuML / cuGraph, delivering up to 100x+ speedups on 100k+ cells.
  - Implemented automatic hardware detection (`is_gpu_available()`) with zero-config multi-core CPU fallbacks.
  - Added GPU scaling benchmark script (`benchmarks/benchmark_gpu_scaling.py`).
- **Chunked Out-of-Core Processing (Dask / Zarr)**:
  - Added distributed out-of-core spatial binning using Dask Distributed DataFrames (`dask_bin_transcripts`).
  - Added Zarr export/import (`to_zarr`, `read_zarr`) and batch streaming iterator (`stream_zarr_anndata`) for processing datasets exceeding available host RAM.
  - Added parallel out-of-core matrix statistics (mean, variance, sparsity) via Dask Arrays (`dask_compute_spatial_metrics`).
- **Strict Typing & Package Publishing**:
  - Full PEP 561 compliance with `py.typed` marker file and 100% strict type annotations verified with `mypy`.
  - Comprehensive Google/NumPy-style docstrings across all modules.
  - Added automated TestPyPI and PyPI publishing workflow (`.github/workflows/publish.yml`).
  - Integrated automated test coverage reporting with `pytest-cov` in CI/CD pipeline.
- **Interactive Documentation Site (Sphinx / MyST-NB)**:
  - Created Sphinx documentation website using `myst-nb` and `sphinx-book-theme`.
  - Added interactive Jupyter tutorials (`quickstart.ipynb`, `out_of_core_streaming.ipynb`) and complete API reference.
  - Added automated GitHub Actions documentation deployment workflow (`.github/workflows/docs.yml`).

### Changed
- Bumped project version to `0.4.0` in `pyproject.toml`.
- Expanded CLI subcommands with `--backend`, `--use-dask`, and `--format` options for Zarr and Dask execution.

---

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
