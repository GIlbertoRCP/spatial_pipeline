# Project Plan: Photo-Spot Finder (working title)

Repurpose `spatial_pipeline` from spatial transcriptomics into a spatial engine for
**geotagged photos**: find where people photograph, what they photograph there, when the
light is best, and which spots are distinctive but under-visited. Start with Colorado, then
scale to the US or the world.

**One-line pitch:** *"Where and when should I shoot?"*, answered from tens of millions of geotagged
photos using image embeddings, spatial statistics, and live weather/sun data.

---

## 1. Why this fits the existing code

The pipeline already does "millions of labeled points → bins → neighbor graph → which labels
cluster spatially". Photos swap in almost directly:

| Current (transcriptomics) | New (photos) | Change needed |
|---|---|---|
| transcript at `(x, y)` | photo at `(lat, lon)` | Project lat/lon to meters before binning (see §4) |
| `gene` column (categorical) | `scene` column: cluster ID from CLIP embeddings, or a Flickr tag | Rename `gene` → `label` throughout |
| bins × genes sparse matrix (`build_sparse_matrix`, `compute_square_bins`) | hexes × scene-types matrix | Rename only |
| `compute_spatially_variable_genes` (global Moran's I / Geary's C) | which scene types are spatially clustered | Rename → `compute_spatially_variable_labels` |
| *(missing)* | **where** each scene type clusters | **New:** local statistics (Getis-Ord Gi\*, local Moran's I / LISA) |
| tiles + halo exchange (`out_of_core.py`) | regional tiles of the US/world | Reused; needs real large data to prove it |
| `stream_zarr_anndata`, Zarr storage | store embeddings (N × 512 floats) out of core | Extend to dense embedding arrays |
| `to_pyg_data` | graph neural network over hexes | Reused in milestone 5 |
| `app.py` (Streamlit) | map UI | Rewrite the views; keep the scaffolding |

**Drop or replace:** Xenium/QV logic in `io.py` (`load_coordinates`, `min_qv`), the
transcript mock generator, AnnData as the main format (keep it optional or remove), and the
biology keywords/classifiers in `pyproject.toml`. Rewrite the README and `PROFILING_REPORT.md`
using only numbers measured on real data.

---

## 2. Data sources

| Source | What it gives | Notes / risks |
|---|---|---|
| **YFCC100M** | ~99M Creative Commons Flickr items, ~48M geotagged, with tags, timestamps, camera EXIF | Ends around 2014. Check current access terms before use. Metadata is small; **downloading images is the expensive part** |
| **Flickr API** | Recent geotagged photos for a bounding box | Needs an API key; rate-limited. Use for recent data and the live layer |
| **Open-Meteo** | Free cloud-cover / weather forecasts | For "shoot tonight" scoring |
| **`astral` (Python)** | Sunrise/sunset, golden hour, blue hour for any lat/lon/date | Local computation, no API |
| **OpenStreetMap** (optional) | Trails, roads, parking, viewpoints | "Is it reachable?" and hidden-gem filters |

**Licensing:** store metadata and embeddings freely. Only show thumbnails with attribution and
link back to Flickr. Never re-host full images.

---

## 3. Architecture

```
ingest/      YFCC metadata (bulk)  ·  Flickr API (incremental)  ·  dedupe by photo ID / owner
embed/       image download (rate-limited, resumable) → CLIP embeddings → Zarr store
label/       cluster embeddings (MiniBatchKMeans / HDBSCAN) → scene types; name them via CLIP text prompts
engine/      project → hex/square bins → bins × labels matrix → spatial graph → global + local stats
time/        per-hex histograms by hour-relative-to-sunset and by month (golden hour, seasons)
score/       hidden-gem score, "shoot tonight" score (hotspot × light × clouds × distance)
serve/       FastAPI (or keep Streamlit) + map frontend (deck.gl / pydeck, or Leaflet)
```

Keep `engine/` free of photo-specific code so it stays a reusable spatial core. Everything
photo-specific lives in adapters around it.

---

## 4. Technical decisions to make early

1. **Coordinates.** Current binning assumes planar `x, y`. Options:
   - Colorado / US: project with `pyproj` to an equal-area CRS (e.g. EPSG:5070, CONUS Albers) and reuse the existing binners as-is.
   - Global: use **H3** hexes instead. They're the industry standard and good to have on a resume, but they replace the hex binner rather than reuse it.
   - Recommendation: projected CRS for milestones 1–4, H3 for the global scale-up.
2. **Bias correction.** A few prolific users dominate photo counts. Count *unique photographers
   per hex* (or cap photos per user per hex per day), not raw photos. Otherwise "hotspots" are
   just where one person lives.
3. **Embedding budget.** Embedding all 48M images means downloading terabytes. Plan:
   embed Colorado fully (a manageable subset), and use metadata/tags only for the global
   big-data benchmark. Embed more only if time and storage allow.
4. **Scene labels.** Start with k-means on embeddings (k ≈ 50–200), then name clusters by
   comparing them to CLIP text prompts ("a waterfall", "aspen trees in fall", "city skyline at
   night"). Hand-check a sample. Bad labels ruin everything downstream.

---

## 5. Milestones

Each one ends with something shippable.

### M1: Colorado hotspot map (metadata only)
- **Data source: iNaturalist** (`fetch-inat`, free, no key). Flickr's API turned out to require a paid Pro account. The project now leans toward wildlife and nature photography: species are the labels, observers the photographers. YFCC100M (`ingest-yfcc`) stays available for landscapes/cities and the M5 scale benchmark.
- Seasonality (M3) becomes a headline feature with species data: when to go, not just where (elk rut, wildflower peaks, migration). **Month and hour profiles with lift are done** (`timing`); golden-hour alignment via sun position is still to do.
- Ingest YFCC metadata, filter to the Colorado bounding box, dedupe, and apply the per-user cap.
- Generalize the engine: `gene` → `label`, lat/lon projection, and real-data loaders replacing the mock generator.
- Hex hotspot map by Flickr tag; global Moran's I to rank which tags cluster spatially.
- **Done when:** a map of Colorado hotspots loads from real data, tests pass on a small real fixture, and the README is rewritten.

### M2: Scene understanding (embeddings)
- Resumable, rate-limited image fetcher, then CLIP embeddings (OpenCLIP) stored in Zarr.
- Clustering into scene types and CLIP-prompt naming.
- Local statistics (Gi\* / LISA) answering "where are the waterfall / alpine-lake / fall-color hotspots?" (**Gi\* done** for tags and photographer density; scene types will plug into the same code)
- **Done when:** you can pick a scene type and see its significant hotspots on the map.

### M3: Hidden gems + timing
- Hidden-gem score: hexes whose embedding centroid is visually distinctive relative to their neighbors, with *low* photographer counts.
- Timing: per-hex golden-hour vs. midday share, month-by-month activity (aspen peak, wildflowers, snow).
- **Done when:** for any spot you can see the best time of day and month, and you can browse a ranked list of gems.
- **Validation:** go shoot 3–5 recommended gems yourself. Those photos go in the README and portfolio.

### M4: Live "shoot tonight" + visual search
- Score = hotspot quality × golden-hour timing (`astral`) × forecast clouds (Open-Meteo) × distance from you.
- Visual search: upload a photo → embed → nearest neighbors (FAISS) → map of similar-looking places.
- Deploy a public demo (Hugging Face Spaces / Fly.io / Render).
- **Done when:** a public URL answers "where should I shoot tonight near Denver?"

### M5: Scale + ML (the big-data and resume layer)
- Run the engine over all ~48M geotagged YFCC records (metadata) with the existing out-of-core tiling, under a fixed RAM cap. Benchmark against DuckDB-spatial and GeoPandas on the same task.
- Optional: graph neural network over the hex graph (`to_pyg_data`) predicting the photogenic score of under-photographed hexes from their neighbors and OSM features. Evaluate against a simple baseline (neighbor average).
- Optional: move hot loops to Rust (PyO3) if profiling shows the Python binning is the bottleneck.
- **Done when:** the README has a real benchmark table and a write-up/blog post explains the design and results.

---

## 6. What goes on the resume

Aim for measured lines like these (fill in real numbers as you go):
- "Built a spatial analytics engine processing **N M geotagged photos** under **X GB RAM**, **Y×** faster than GeoPandas/DuckDB on hex aggregation."
- "Clustered **N** CLIP image embeddings into scene types; detected statistically significant hotspots with Getis-Ord Gi\*."
- "Deployed a live 'where to shoot tonight' service combining spatial hotspots, sun position and weather forecasts."
- Link: public demo + write-up + your own photos taken at recommended spots.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Image download volume / rate limits | Colorado-only embeddings; resumable fetcher; metadata-only for global scale |
| YFCC data is old (≤2014) | Supplement with Flickr API; frame YFCC as the historical base |
| Prolific-user bias | Unique-photographer counts / per-user caps (§4.2) |
| Bad scene clusters | Hand-check samples; tune k; allow merging clusters |
| Scope creep | Ship M1–M3 before touching ML or Rust |
