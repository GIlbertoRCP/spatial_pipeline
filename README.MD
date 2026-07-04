# The Sparse Matrix and Spatial Binning Pipeline

This repository contains a high-performance Python spatial transcriptomics coordinate preprocessing pipeline. It is optimized to perform segmentation-free spatial binning (on standard square grids or hexagonal meshes) on millions of molecules, aggregate expression counts into Compressed Sparse Row (CSR) or Compressed Sparse Column (CSC) matrices, and build spatial adjacency graphs using KD-Trees.

## Project Structure

```
spatial_pipeline/
|
+-- src/
|   L-- spatial_pipeline/
|       +-- __init__.py
|       +-- io.py            # Fast coordinate loading
|       +-- binning.py       # Square and Hexagonal binning logic
|       +-- graphs.py        # KD-Tree adjacency graph construction
|
+-- tests/
|   +-- __init__.py
|   +-- test_binning.py      # Unit tests for spatial math
|
+-- pyproject.toml           # Build system & dependencies
+-- README.md                # Clear project documentation
+-- run_pipeline.py          # CLI Entry point
```

## Scientific and Mathematical Background

### 1. Square Grid Binning
The square grid divides the continuous 2D space into uniform square cells of side length $D$.
For a point $(x, y)$, the integer grid bin index is computed as:
$$bin_x = \lfloor \frac{x - x_{min}}{D} \rfloor$$
$$bin_y = \lfloor \frac{y - y_{min}}{D} \rfloor$$
Where $x_{min}$ and $y_{min}$ are the minimum coordinate bounds of the dataset.

The Cartesian coordinates of the bin center are computed as:
$$x_{center} = (bin_x + 0.5) \cdot D + x_{min}$$
$$y_{center} = (bin_y + 0.5) \cdot D + y_{min}$$

---

### 2. Hexagonal Grid Binning
Hexagonal binning utilizes pointy-topped regular hexagons. This grid style offers isotropic neighborhood properties (distance to all 6 surrounding neighbors is identical).
- Center-to-center distance is defined as $D$.
- Outer radius of each hexagon is $R = \frac{D}{\sqrt{3}}$.

#### Coordinate Inversion
We map Cartesian coordinates $(x, y)$ to fractional axial coordinates $(q_{frac}, r_{frac})$:
$$q_{frac} = \frac{\sqrt{3}x - y}{3R}$$
$$r_{frac} = \frac{2}{3} \frac{y}{R}$$
We define the redundant coordinate $s_{frac} = -q_{frac} - r_{frac}$ to preserve the constraint $q + r + s = 0$.

#### Axial Hex Rounding
To identify which discrete hexagon a point falls into, we round each component to the nearest integer:
$$q_{round} = \lfloor q_{frac} + 0.5 \rfloor$$
$$r_{round} = \lfloor r_{frac} + 0.5 \rfloor$$
$$s_{round} = \lfloor s_{frac} + 0.5 \rfloor$$

We resolve rounding conflicts by adjusting the coordinate with the largest rounding error:
$$\Delta q = |q_{round} - q_{frac}|, \quad \Delta r = |r_{round} - r_{frac}|, \quad \Delta s = |s_{round} - s_{frac}|$$
- If $\Delta q \ge \Delta r$ and $\Delta q \ge \Delta s$: $q_{round} = -r_{round} - s_{round}$
- Else if $\Delta r \ge \Delta s$: $r_{round} = -q_{round} - s_{round}$
- Else: $s_{round} = -q_{round} - r_{round}$

#### Reconstructing Centers
The center coordinates of a hexagon cell defined by axial indices $(q, r)$ are:
$$x_{center} = R \cdot (\sqrt{3}q + \frac{\sqrt{3}}{2}r)$$
$$y_{center} = R \cdot (1.5r)$$

---

### 3. Sparse Count Matrix Assembly
For large spatial transcriptomics datasets (like Xenium, MERFISH, or CosMx), keeping coordinates in dense formats leads to severe memory overhead. By aggregating coordinate counts into a sparse matrix, we store only non-zero occurrences.
- Standard COO construction is utilized. The row index represents the sequential active bin ID, and the column index represents the unique gene ID.
- Scipy's `coo_matrix` constructor automatically sums values at duplicate coordinates during conversions to `csr_matrix` or `csc_matrix`, offering a vectorized aggregation process in optimized C.

---

### 4. KD-Tree Spatial Adjacency Graphs
To analyze spatial context or run graph algorithms (like GNNs), we build a spatial adjacency graph where nodes represent active bins and edges connect adjacent bins.
- A `scipy.spatial.KDTree` is built on the active bin centers.
- We query the tree for pairs within a distance threshold $T$:
  - Square Grid (4-connectivity): $T = 1.01 \cdot D$
  - Square Grid (8-connectivity): $T = 1.01 \cdot D \cdot \sqrt{2}$
  - Hexagonal Grid: $T = 1.01 \cdot D$
- The output graph is structured as a symmetric CSR sparse adjacency matrix.

---

## Installation

Ensure you have Python 3.10 or later installed. Install the package and dependencies in editable mode:

```bash
pip install -e .
```

To install development dependencies (for testing):

```bash
pip install -e ".[test]"
```

## Running the Pipeline CLI

The main CLI is located in `run_pipeline.py`.

### 1. Run with Synthetic Mock Data (Default Resolution)
Generate a realistic mock coordinate dataset (150,000 molecules) and run the full pipeline:
```bash
python run_pipeline.py --generate-mock --bin-type square --resolution 20.0
```

To run with hexagonal binning:
```bash
python run_pipeline.py --generate-mock --bin-type hex --resolution 20.0
```

### 2. Download and Run Public 10x Genomics Xenium Data
This command downloads a public 2 FOVs Xenium human breast tissue dataset (approx. 379 MB), extracts it, and processes it:
```bash
python run_pipeline.py --download --bin-type hex --resolution 20.0
```

### 3. Process an Existing Dataset File
Run the pipeline on your own custom coordinate file:
```bash
python run_pipeline.py --input /path/to/transcripts.csv.gz --bin-type square --resolution 10.0
```

### CLI Parameters
- `--input`: Path to input transcript coordinate file.
- `--bin-type`: Choice of grid pattern, either `square` or `hex` (default: `square`).
- `--resolution`: Grid spacing resolution (default: `20.0` microns).
- `--min-qv`: Minimum transcript quality value filtering threshold (default: `20.0`).
- `--output-dir`: Folder path to save the generated numpy and sparse matrix outputs (default: `output`).
- `--generate-mock`: Flag to run with a synthetic mock dataset.
- `--download`: Flag to download and process the official 10x Genomics sample.

### Outputs
Outputs are saved in the designated `--output-dir` (default: `output/`):
- `expression_matrix.npz`: Binned counts in Compressed Sparse Row format.
- `adjacency_matrix.npz`: Spatial adjacency matrix in Compressed Sparse Row format.
- `bin_centers.npy`: NumPy array of $(x, y)$ Cartesian coordinates of the active bin centers.
- `bin_coords.npy`: NumPy array of integer coordinates (bin indices/axial coordinates).
- `genes.npy`: Array of unique gene symbols corresponding to sparse matrix columns.
- `offset.npy` (For square grids only): Base coordinates offset $(x_{min}, y_{min})$.
- `spatial_binned_plot.png`: Spatial visualization of binned transcript densities.

## Running Tests
Run pytest to verify the correctness of the spatial math and structures:

```bash
pytest
```
