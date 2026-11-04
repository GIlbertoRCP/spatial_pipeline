# Quickstart

This walk-through goes from nothing to a hotspot map of one month of Denver-area wildlife and
plant photos in about five minutes. It needs no API key.

## 1. Fetch observations

```bash
spatial-pipeline fetch-inat --bbox denver --observed-from 2025-06-01 --observed-to 2025-06-30 --output photos.parquet
```

That's about 35 requests to the iNaturalist API (a few minutes at one request per second). The
default filters keep research-grade observations with Creative Commons photos, public coordinates
and a location uncertainty under 1 km. Each page is cached in `.inat_cache/`, so rerunning the
command resumes instead of starting over.

```text
| Observations kept       | 6,417 |
| Observers               | 954   |
| Species / taxa          | 1,046 |
```

## 2. Compute hotspots

```bash
spatial-pipeline hotspots --input photos.parquet --output-dir hotspots --min-label-photographers 8
```

This bins the photos into 1 km hexagons, ranks subjects by how strongly they cluster in space, and
finds each subject's significant hot spots. It writes these files to `hotspots/`:

| File | Contents |
| :--- | :--- |
| `hexes.parquet` | One row per hexagon: center, photo and photographer counts, density *Gi\** |
| `label_counts.parquet` | Photographers per (hexagon, subject) |
| `local_hotspots.parquet` | Significant *Gi\** hot spots per (hexagon, subject) |
| `ranking.csv` | Subjects ranked by Moran's *I*, with FDR p-values, hot-spot counts and peak months |
| `timing.parquet` | Month and hour profiles per subject, with lift |
| `monthly_counts.parquet` | Photographers per (hexagon, subject, month) |
| `photo_samples.parquet` | A few credited thumbnails per (hexagon, subject) |
| `subjects.parquet` | Group and scientific name per subject |
| `meta.json` | Resolution, projection, counts and run settings |

## 3. Explore the map

```bash
spatial-pipeline ui
```

Pick a subject, for example *black-tailed prairie dog*, and turn on **Only significant hot
spots**. Click a hexagon to see that spot's top subjects, photos and a directions link. See
{doc}`dashboard` for the full tour.

## The same in Python

```python
from spatial_pipeline import DENVER_METRO_BBOX, INatClient, INatFetcher, compute_hotspots
from spatial_pipeline.timing import compute_timing, describe_peaks

fetcher = INatFetcher(INatClient(), cache_dir=".inat_cache")
query = INatFetcher.build_query(DENVER_METRO_BBOX, observed_from="2025-06-01", observed_to="2025-06-30")
photos = fetcher.fetch(query)

result = compute_hotspots(photos, resolution=1000.0, min_label_photographers=8)
print(result.ranking[["label", "morans_i", "fdr_p_value", "hotspot_hexes"]].head(10))

timing = compute_timing(photos, labels=result.labels)
print(describe_peaks(timing, "common whitetail"))   # e.g. 'Jun; 17:00, 16:00, 18:00'
```
