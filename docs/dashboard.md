# Dashboard

```bash
spatial-pipeline ui
```

The dashboard is a Streamlit app with a deck.gl map. It reads a folder written by
`spatial-pipeline hotspots`: `hotspots/` by default, the `HOTSPOTS_DIR` environment variable, or
the folder set under **Data source** in the sidebar.

## Sidebar

- **When:** *All year* or a single month. With a month picked, the map shows only that month's
  photographers, and subjects are ordered by what is photographed most that month. ↑2.1× marks
  subjects photographed unusually often then (lift).
- **Group:** birds, mammals, plants, insects and other groups present in the data.
- **What do you want to photograph?:** subjects that cluster in specific places, shown with
  scientific names.
- **Only significant hot spots:** keep only hexagons that are significant *Gi\** hot spots.
- **Minimum photographers per hex:** hide quiet hexagons.

## Map and spot panel

Hexagon colors follow a log scale from sage (few photographers) to sunset orange (many), and
quiet hexagons are partly transparent so the basemap shows through. Hover for counts and hot-spot
status.

**Click a hexagon** to open the spot panel:

- photographers and photos, with a directions link
- a summary of groups (`insects 19 · birds 16 · plants 13`)
- the top subjects there, ★ marking a significant hot spot
- photos from that spot, linking to the original observation, with photographer credit

The selection is outlined on the map and kept in the URL (`?hex=563`), so a spot can be shared
as a link.

With no spot selected, the panel shows the subject's best months and hours, photos from its top
spots and, when a month is picked, an **In season** list of one-click suggestions.

## When to go and tables

Below the map:

- month and hour-of-day bar charts for the selected subject
- the top spots, with hot-spot z-scores and map links
- the most place-specific subjects, with clustering strength, hot-spot counts and peak months

## Theme

The theme lives in `.streamlit/config.toml`. `spatial-pipeline ui` runs from the repository folder
so Streamlit finds it.
