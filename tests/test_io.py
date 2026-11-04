import pandas as pd
import pytest

from spatial_pipeline.io import load_coordinates


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_load_coordinates(tmp_path, suffix):
    points = pd.DataFrame({"x": [1, 2, 3], "y": [4.5, 5.5, 6.5], "label": ["elk", "elk", "mallard"], "extra": [0, 0, 0]})
    path = str(tmp_path / f"points{suffix}")
    if suffix == ".csv":
        points.to_csv(path, index=False)
    else:
        points.to_parquet(path, index=False)

    df = load_coordinates(path)
    assert list(df.columns) == ["x", "y", "label"]
    assert df["x"].dtype == "float64"
    assert isinstance(df["label"].dtype, pd.CategoricalDtype)
    assert set(df["label"].cat.categories) == {"elk", "mallard"}


def test_load_coordinates_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_coordinates(str(tmp_path / "nope.csv"))
