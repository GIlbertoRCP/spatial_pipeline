import os
import tempfile
import pytest
from typer.testing import CliRunner

from spatial_pipeline.cli import app
from spatial_pipeline.io import generate_vectorized_mock_data

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Spatial Pipeline CLI" in result.stdout


def test_cli_run():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_csv = tmp.name
        
    try:
        generate_vectorized_mock_data(tmp_csv, num_molecules=300)
        result = runner.invoke(app, ["run", "--input", tmp_csv, "--resolution", "10.0"])
        assert result.exit_code == 0
        assert "Execution Summary" in result.stdout
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)


def test_cli_svg():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_csv = tmp.name
        
    try:
        generate_vectorized_mock_data(tmp_csv, num_molecules=300)
        result = runner.invoke(app, ["svg", "--input", tmp_csv, "--top-n", "5"])
        assert result.exit_code == 0
        assert "Top 5 Spatially Variable Genes" in result.stdout
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)


def test_cli_export():
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_csv = os.path.join(tmp_dir, "input.csv")
        tmp_h5ad = os.path.join(tmp_dir, "output.h5ad")
        generate_vectorized_mock_data(tmp_csv, num_molecules=300)
        
        result = runner.invoke(app, ["export", "--input", tmp_csv, "--output", tmp_h5ad])
        assert result.exit_code == 0
        assert os.path.exists(tmp_h5ad)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_cli_benchmark():
    result = runner.invoke(app, ["benchmark", "--num-molecules", "400"])
    assert result.exit_code == 0
    assert "Binning Algorithm Tradeoffs" in result.stdout
