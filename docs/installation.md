# Installation Guide

`spatial_pipeline` supports standard CPU environments as well as NVIDIA CUDA-accelerated setups with RAPIDS cuGraph and cuML.

---

## 1. Standard Installation (CPU)

To install `spatial_pipeline` with default CPU dependencies:

```bash
pip install spatial_pipeline
```

To install from source:

```bash
git clone https://github.com/gilbertorcp/spatial_pipeline.git
cd spatial_pipeline
pip install -e .
```

---

## 2. GPU Acceleration (RAPIDS cuGraph / cuML)

For GPU acceleration on NVIDIA Linux systems, install the RAPIDS 24+ ecosystem:

```bash
pip install spatial_pipeline[gpu]
```

Or via Conda/Mamba:

```bash
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=24.08 cugraph=24.08 cupy cuda-version=12.5
```

---

## 3. Optional Feature Extras

You can install optional dependency groups based on your workflow:

```bash
# Documentation building dependencies
pip install spatial_pipeline[docs]

# Testing & Type Checking dependencies
pip install spatial_pipeline[test]

# Complete installation with all optional features
pip install spatial_pipeline[all]
```
