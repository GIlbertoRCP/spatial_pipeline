# Installation

`spatial_pipeline` needs Python 3.10 or newer. It runs on Windows, macOS and Linux, and on a CPU
only. NVIDIA GPU acceleration is optional.

## From source

```bash
git clone https://github.com/GIlbertoRCP/spatial_pipeline.git
cd spatial_pipeline
pip install -e .
```

This installs the `spatial-pipeline` command and the Python package.

## Optional extras

```bash
pip install -e .[test]   # pytest, coverage, mypy
pip install -e .[docs]   # Sphinx, MyST-NB and the book theme, to build this site
pip install -e .[gpu]    # RAPIDS cuML / cuGraph / CuPy (Linux with an NVIDIA GPU)
```

The GPU extra only speeds up spatial graph construction on very large inputs (see
{doc}`gpu_acceleration`). Everything else runs the same without it.

## Check the install

```bash
spatial-pipeline --help
pytest
```

## Build the documentation locally

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.
