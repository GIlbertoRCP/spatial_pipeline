# The Sparse Matrix & Spatial Binning Pipeline

A high-performance Python package and interactive web visualizer for spatial transcriptomics data processing.

**Performance Highlight**: Processes millions of continuous spatial coordinates into discrete topological graphs and sparse matrices in seconds.

---

## What is Spatial Transcriptomics?

In simple terms, traditional genomics (like bulk or single-cell RNA sequencing) is like placing a tissue slice in a blender: you find out *what* genes are present, but you lose all information about *where* they were located.

**Spatial Transcriptomics** is a revolutionary imaging technology (used in platforms like 10x Genomics Xenium, Vizgen MERSCOPE, and NanoString CosMx) that acts as a high-resolution map. It records the exact $(x, y)$ coordinates of millions of individual RNA molecules directly inside a tissue slice. This allows scientists to see exactly which genes are active in different cellular neighborhoods.

---

## Why This Pipeline is Needed

With millions of raw coordinates per tissue sample, processing this data using standard Python loops would crash your system or take hours. 

This pipeline resolves this computational bottleneck using **NumPy vectorization** and **SciPy sparse structures**:
1. **Vectorized Spatial Binning**: Fast coordinate binning that segments continuous space into discrete square grids or pointy-topped hexagonal grid meshes.
2. **Sparse Expression Matrix Export**: Aggregates raw transcript counts inside each spatial bin into memory-efficient Compressed Sparse Row (CSR) matrices.
3. **KD-Tree Graph Networks**: Connects adjacent bins into a spatial adjacency network using a fast KD-Tree search.
4. **Graph Laplacian Construction**: Automatically generates the Graph Laplacian matrix ($L = D - A$) to enable immediate downstream spatial analyses (such as Graph Neural Networks or spatial variance analysis).

---

## Project Structure

```
spatial_pipeline/
|
+-- src/
|   L-- spatial_pipeline/
|       +-- __init__.py
|       +-- io.py            # Fast coordinate loading & low-memory dtypes
|       +-- binning.py       # Vectorized Square Grid & Hexagonal Grid math
|       +-- graphs.py        # KD-Tree spatial adjacency graph & Graph Laplacian
|
+-- tests/
|   +-- __init__.py
|   +-- test_binning.py      # Unit tests for coordinate binning & graphs
|
+-- pyproject.toml           # Package requirements & pytest configuration
+-- README.MD                # Simple, clear project documentation
+-- run_pipeline.py          # Main CLI entry point with mock data generator
+-- app.py                   # Streamlit + Plotly interactive Web Dashboard
```

---

## Installation

Ensure you have Python 3.10 or later installed, then install the package in editable mode:

```bash
pip install -e .
```

To install development dependencies (for testing):

```bash
pip install -e ".[test]"
```

---

## Running the Pipeline

### 1. Command Line Interface (CLI)

The CLI tool `run_pipeline.py` executes data loading, binning, and graph calculations.

#### Run Out-of-the-Box with Mock Data (For Recruiters/Testing)
You do not need to download massive biological files to test the project. Use the `--generate-mock` flag to instantly generate a synthetic dataset and run the pipeline:

```bash
python run_pipeline.py --generate-mock --resolution 10.0 --max_dist 15.0
```

#### Run with Custom Coordinates File
To process your own spatial transcriptomics coordinates CSV file (expecting columns `x`, `y`, `gene`):

```bash
python run_pipeline.py --input path/to/coordinates.csv --resolution 10.0 --max_dist 15.0
```

### 2. Interactive Web Dashboard

To launch the interactive Streamlit + Plotly dashboard locally:

```bash
streamlit run app.py
```

The web dashboard allows you to:
- Upload your own coordinate CSV/compressed files.
- Switch parameters (resolution grid width, graph connection distance) in real-time.
- View interactive, WebGL-accelerated scatter maps of molecule clusters.
- Inspect the spatial adjacency graph overlay connecting adjacent grid boxes.
- Download the generated binned counts (CSV), adjacency edge list (CSV), and Graph Laplacian (NPZ) matrices.

---

## Running Tests

Verify the spatial binning math and graph Laplacian calculations by running:

```bash
pytest
```
