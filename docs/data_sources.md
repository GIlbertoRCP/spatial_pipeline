# Data sources

Every source produces the same **photo table**, one row per photo, so everything downstream works
on any of them:

| Column | Meaning |
| :--- | :--- |
| `photo_id`, `user_id` | Photo and photographer identifiers |
| `lon`, `lat` | WGS84 coordinates |
| `taken_at` | Local wall-clock time the photo was taken |
| `tags` | Comma-separated subjects (species and group for iNaturalist, user tags for Flickr) |
| `thumbnail_url`, `page_url`, `owner_name`, `attribution`, `license` | For showing photos with credit |

## iNaturalist (recommended)

```bash
spatial-pipeline fetch-inat --bbox colorado --output photos.parquet
```

- **No key needed.** Each observation becomes a row. The observer counts as the photographer,
  and the tags are the species' common name plus its group (`birds`, `mammals`, `plants`, ...).
- **Defaults** keep research-grade observations (identifications confirmed by the community) with
  Creative Commons photos. That's about 1.2 million observations for Colorado. Use `--quality all`
  or `--all-licenses` to widen it.
- **Blurred locations are excluded.** iNaturalist randomizes the public coordinates of threatened
  species and of users who choose privacy, and these would create false hot spots. Observations
  with a location uncertainty above `--max-accuracy-m` (default 1,000 m) are dropped too.
- **Polite fetching.** Normal paging stops at 10,000 results, so the fetcher pages by observation
  id (`id_above`), which has no cap. It sends about one request per second and stops after
  `--max-requests` (default 9,000) to stay under iNaturalist's guideline of ~10,000 requests per
  day.
- **Resumable.** Each page is cached in `--cache-dir`. Rerun the same command to continue. The cache
  is tied to the query, so a different query needs a different cache directory.

All of Colorado takes several hours spread over a couple of runs. A city and a single month
takes a few minutes.

## YFCC100M (historic Flickr photos, up to 2014)

```bash
spatial-pipeline ingest-yfcc --input yfcc100m_dataset.sql --output photos.parquet --bbox colorado
```

The Yahoo Flickr Creative Commons 100M dataset covers landscapes and cities that iNaturalist
doesn't. The metadata is a single SQLite database in the public AWS Multimedia Commons bucket
(`s3://multimedia-commons/tools/etc/yfcc100m_dataset.sql`, about 65.6 GB). The original
tab-separated release (optionally `.bz2`) works too, and 23- and 25-column layouts are detected
automatically. Rows are streamed in chunks (SQLite pre-filters by bounding box), so the full
dataset never has to fit in memory.

## Flickr API (requires Flickr Pro)

`fetch-flickr` is a tested, resumable fetcher. Flickr returns at most about 4,000 results per
query, so it splits upload-date windows, then areas, until each window fits. Flickr now issues
API keys only to paid Pro members, so this is kept for anyone who has one. The key is read from
the `FLICKR_API_KEY` environment variable.

## Regions

`--bbox` accepts `colorado`, `denver`, or `min_lon,min_lat,max_lon,max_lat`.

## Licensing

iNaturalist photos fetched with the default filters carry Creative Commons licenses. Show them
with the `attribution` text and a link to `page_url`, as the dashboard does. Review each source's
terms of use before publishing anything built from the data.
