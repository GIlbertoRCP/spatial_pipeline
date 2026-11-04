# GPU acceleration (optional)

Building spatial graphs (fixed-radius or k-nearest-neighbor) is the step that grows fastest with
input size. `spatial_pipeline` can hand it to an NVIDIA GPU through RAPIDS cuML, and falls back
to the CPU automatically.

```bash
pip install -e .[gpu]   # Linux with an NVIDIA GPU and CUDA 12
```

## Usage

```python
import spatial_pipeline as sp

sp.is_gpu_available()   # True when CuPy + cuML/cuGraph import and a CUDA device is present

# Radius graph: GPU when available and the input is large (>= 5,000 bins), CPU otherwise
adjacency, laplacian = sp.build_spatial_graph(centers, max_distance=1050.0, backend="auto")

# k-NN graph, forcing the GPU path (falls back to CPU with a warning if it fails)
adjacency, laplacian = sp.build_knn_graph(centers, n_neighbors=6, backend="cuml")
```

CPU backends are `"cpu"` (single process), `"threading"` and `"multiprocessing"` (pass
`n_jobs=-1` for all cores).

## When it matters

For a region like Colorado at 1 km hexagons, the graph has tens of thousands of nodes and builds
in well under a second on a CPU, so the GPU path is not needed. It becomes relevant at national or
global scale with fine hexagons. No GPU benchmark has been run for the photo workloads yet; that is
part of the planned scale-up milestone.
