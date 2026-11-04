# Installation

`spatial_pipeline` needs Python 3.10 or newer. It runs on Windows, macOS and Linux, and on a CPU
only. NVIDIA GPU acceleration is optional.

## From source

```bash
git clone https://github.com/gilbertorcp/spatial_pipeline.git
cd spatial_pipeline
pip install -e .
```

This installs the `spatial-pipeline` command and the Python package: fetching, ingestion and the
hotspot statistics. The map and the large-data tools are extras.

## Optional extras

```bash
pip install -e .[ui]     # Streamlit and pydeck, for the map (`spatial-pipeline ui`)
pip install -e .[scale]  # Dask, for out-of-core binning of very large point files
pip install -e .[test]   # pytest, coverage, mypy
pip install -e .[docs]   # Sphinx, MyST-NB and the book theme, to build this site
pip install -e .[gpu]    # RAPIDS cuML / cuGraph / CuPy (Linux with an NVIDIA GPU)
```

The GPU extra only speeds up spatial graph construction on very large inputs (see
{doc}`gpu_acceleration`). Everything else runs the same without it.

## Check the install

```bash
spatial-pipeline --help
pip install -e .[test,ui,scale]
pytest
```

## Build the documentation locally

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.
