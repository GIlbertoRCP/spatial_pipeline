"""
Synthetic YFCC100M metadata in the real file layout, for tests.

Rows cluster around real Colorado landmarks with landmark-specific tags, plus a diffuse
'colorado' tag everywhere, one prolific uploader, and rows the loader must drop
(outside the bbox, videos, missing geotags, low accuracy, duplicates).
"""

from __future__ import annotations

import sqlite3
from urllib.parse import quote_plus

import numpy as np

from spatial_pipeline.photos import YFCC_FIELDS_23, YFCC_FIELDS_25

# name -> (lon, lat, tags)
LANDMARKS = {
    "maroon_bells": (-106.9406, 39.0986, ["maroon bells", "aspen", "lake"]),
    "red_rocks": (-105.2057, 39.6654, ["red rocks", "amphitheatre", "concert"]),
    "garden_of_gods": (-104.8786, 38.8784, ["garden of the gods", "sandstone"]),
    "denver_downtown": (-104.9903, 39.7392, ["denver", "skyline", "city"]),
}

PROLIFIC_USER = "99999999@N00"
PROLIFIC_SPOT = (-105.5000, 40.5000)  # one backyard, far from the landmarks


def _row(photo_id: int, user: str, lon: float | str, lat: float | str, tags: list[str],
         accuracy: int = 16, marker: int = 0, taken: str = "2012-07-14 19:42:10.0") -> dict[str, str]:
    encoded_tags = ",".join(quote_plus(t) for t in tags)
    return {
        "photo_id": str(photo_id),
        "user_id": user,
        "user_nickname": user.split("@")[0],
        "date_taken": taken,
        "date_uploaded": "1342300930",
        "capture_device": "Canon+EOS+5D",
        "title": "IMG_" + str(photo_id),
        "description": "",
        "user_tags": encoded_tags,
        "machine_tags": "",
        "longitude": f"{lon}" if isinstance(lon, str) else f"{lon:.6f}",
        "latitude": f"{lat}" if isinstance(lat, str) else f"{lat:.6f}",
        "accuracy": str(accuracy),
        "page_url": f"http://www.flickr.com/photos/{user}/{photo_id}/",
        "download_url": f"http://farm1.staticflickr.com/1/{photo_id}_abc.jpg",
        "license_name": "Attribution License",
        "license_url": "http://creativecommons.org/licenses/by/2.0/",
        "server_id": "1",
        "farm_id": "1",
        "secret": "abc",
        "secret_original": "def",
        "extension_original": "jpg",
        "marker": str(marker),
    }


def make_rows(seed: int = 0, photographers_per_landmark: int = 30) -> list[dict[str, str]]:
    """Builds the fixture rows. Returns dicts keyed by 23-field YFCC names."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    pid = 1000

    for li, (lon0, lat0, tags) in enumerate(LANDMARKS.values()):
        for u in range(photographers_per_landmark):
            user = f"{1000 + li * 100 + u}@N0{li}"
            for _ in range(int(rng.integers(1, 4))):
                lon = lon0 + rng.normal(0, 0.002)
                lat = lat0 + rng.normal(0, 0.002)
                rows.append(_row(pid, user, lon, lat, tags + ["colorado", "2012", "iphone"]))
                pid += 1

    # Diffuse 'colorado' + 'mountains' photos across the state.
    for u in range(120):
        lon = rng.uniform(-108.8, -102.3)
        lat = rng.uniform(37.1, 40.9)
        rows.append(_row(pid, f"{5000 + u}@N05", lon, lat, ["colorado", "mountains"]))
        pid += 1

    # One prolific uploader: 300 photos from one spot.
    for _ in range(300):
        lon = PROLIFIC_SPOT[0] + rng.normal(0, 0.0005)
        lat = PROLIFIC_SPOT[1] + rng.normal(0, 0.0005)
        rows.append(_row(pid, PROLIFIC_USER, lon, lat, ["backyard", "colorado"]))
        pid += 1

    # Rows the loader must drop.
    rows.append(_row(pid, "1@N09", -122.4194, 37.7749, ["sanfrancisco"]))  # outside bbox
    rows.append(_row(pid + 1, "2@N09", -105.2057, 39.6654, ["red rocks"], marker=1))  # video
    rows.append(_row(pid + 2, "3@N09", "", "", ["nogeo"]))  # not geotagged
    rows.append(_row(pid + 3, "4@N09", -105.2057, 39.6654, ["red rocks"], accuracy=6))  # low accuracy
    rows.append(dict(rows[0]))  # duplicate photo id
    return rows


def write_tsv(path: str, width: int = 25, rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Writes rows as a headerless TSV with the 23- or 25-column YFCC layout."""
    rows = rows if rows is not None else make_rows()
    fields = YFCC_FIELDS_25 if width == 25 else YFCC_FIELDS_23
    lines = []
    for i, row in enumerate(rows):
        full = dict(row, line_number=str(i), photo_hash=f"{i:032x}")
        lines.append("\t".join(full[f] for f in fields))
    if path.endswith(".bz2"):
        import bz2
        with bz2.open(path, "wt", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    else:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    return rows


def write_sqlite(path: str, rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Writes rows to a SQLite table with 25 generically named text columns."""
    rows = rows if rows is not None else make_rows()
    conn = sqlite3.connect(path)
    cols = [f"c{i}" for i in range(25)]
    conn.execute(f"CREATE TABLE yfcc100m_dataset ({', '.join(c + ' TEXT' for c in cols)})")
    data = []
    for i, row in enumerate(rows):
        full = dict(row, line_number=str(i), photo_hash=f"{i:032x}")
        data.append(tuple(full[f] for f in YFCC_FIELDS_25))
    conn.executemany(f"INSERT INTO yfcc100m_dataset VALUES ({', '.join('?' * 25)})", data)
    conn.commit()
    conn.close()
    return rows
