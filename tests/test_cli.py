from typer.testing import CliRunner

from spatial_pipeline.cli import app

runner = CliRunner()


def test_cli_help_lists_photo_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Spatial Pipeline CLI" in result.stdout
    for command in ("fetch-inat", "ingest-yfcc", "fetch-flickr", "hotspots", "ui"):
        assert command in result.stdout
