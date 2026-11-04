# Photo Hotspots

**Where and when should I go to photograph elk, bighorn sheep or columbines?**

`spatial_pipeline` finds wildlife and nature photo hotspots in millions of geotagged photos. It
pulls photographed observations from [iNaturalist](https://www.inaturalist.org) (free, no API key)
or historic Flickr photos from YFCC100M, bins them into equal-area hexagons, and counts **unique
photographers** rather than raw photos. Spatial statistics then show which subjects cluster in specific
places (Moran's *I*), where their significant hot spots are (Getis-Ord *Gi\**), and when they are
photographed (month and hour profiles with lift).

On one month of real Denver-area observations (June 2025: 6,417 observations from 954 observers),
the significant hot spots land where a naturalist would point you. Black-tailed prairie dogs
cluster on the Rocky Mountain Arsenal refuge, mallards along the South Platte and city lakes, and
pinewoods geranium in the western foothills.

---

## Quick example

```bash
pip install -e .
spatial-pipeline fetch-inat --bbox denver --observed-from 2025-06-01 --observed-to 2025-06-30 --output photos.parquet
spatial-pipeline hotspots --input photos.parquet --output-dir hotspots
spatial-pipeline ui
```

```python
from spatial_pipeline import DENVER_METRO_BBOX, INatClient, INatFetcher, compute_hotspots

fetcher = INatFetcher(INatClient(), cache_dir=".inat_cache")
photos = fetcher.fetch(INatFetcher.build_query(DENVER_METRO_BBOX, observed_from="2025-06-01"))
result = compute_hotspots(photos, resolution=1000.0)

result.ranking.head(10)   # subjects ranked by spatial clustering (Moran's I, FDR-corrected)
result.local_hotspots     # significant Gi* hot spots per (hex, subject)
```

---

## What's inside

| Piece | What it does |
| :--- | :--- |
| Data sources | iNaturalist fetcher (id-based paging, privacy and accuracy filters, resumable), YFCC100M reader (TSV / SQLite), Flickr fetcher (needs a Pro key) |
| Hotspots | Hex binning on an equal-area projection, photographer-weighted counts, Moran's *I* ranking, Getis-Ord *Gi\** hot spots, Benjamini-Hochberg FDR |
| Timing | Month and hour-of-day profiles per subject, with lift against all photos |
| Dashboard | Streamlit + deck.gl map: month and group filters, click-a-spot details, credited photos |
| Engine | Generic labeled-point binning, CPU/GPU spatial graphs, out-of-core tiling and Dask / Zarr streaming |

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: Guides

data_sources
methods
dashboard
```

```{toctree}
:maxdepth: 2
:caption: Engine

out_of_core_dask_zarr
gpu_acceleration
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
