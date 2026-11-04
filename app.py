"""
Streamlit map of photo hotspots produced by `spatial-pipeline hotspots`.

Run with `spatial-pipeline ui` or `streamlit run app.py` from the repository root
(so `.streamlit/config.toml` supplies the theme).
"""

from __future__ import annotations

import html
import os

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from spatial_pipeline.hotspots import hex_polygons_lonlat, load_hotspots
from spatial_pipeline.inat import ICONIC_GROUPS
from spatial_pipeline.timing import ALL_LABEL, MONTH_NAMES, describe_peaks

st.set_page_config(page_title="Photo Hotspots", page_icon="📷", layout="wide")

ALL_PHOTOS = ALL_LABEL
GROUP_TAGS = set(ICONIC_GROUPS.values())  # broad iNaturalist groups ('birds', 'plants', ...)
ACCENT = "#E0A458"
# Low -> high photographer counts: muted sage, golden-hour amber, sunset orange.
RAMP = np.array([[62, 92, 88], [224, 164, 88], [255, 107, 61]], dtype=np.float64)
RAMP_CSS = "linear-gradient(90deg, rgb(62,92,88), rgb(224,164,88), rgb(255,107,61))"

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 2rem; }}
    [data-testid="stMetricValue"] {{ font-size: 1.7rem; }}
    .legend {{ display: flex; align-items: center; gap: .6rem; font-size: .8rem; opacity: .85; margin: .3rem 0 0; }}
    .legend .bar {{ flex: 0 0 180px; height: 10px; border-radius: 5px; background: {RAMP_CSS}; }}
    .photo-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap: 8px; margin: .4rem 0 1rem; }}
    .photo-grid a {{ text-decoration: none; color: inherit; }}
    .photo-grid img {{ width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px; display: block;
                       border: 1px solid rgba(255,255,255,.08); }}
    .photo-grid span {{ display: block; font-size: .68rem; opacity: .7; margin-top: 2px;
                        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .spot-facts {{ font-size: .92rem; line-height: 1.6; }}
    .badge {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .75rem;
              background: rgba(224,164,88,.18); color: {ACCENT}; border: 1px solid rgba(224,164,88,.4); }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def _load(out_dir: str) -> dict[str, object]:
    return load_hotspots(out_dir)


@st.cache_data
def _polygons(x: np.ndarray, y: np.ndarray, resolution: float, crs: str) -> list[list[list[float]]]:
    return hex_polygons_lonlat(x, y, resolution, crs)


def _colors(values: np.ndarray) -> np.ndarray:
    """Maps values onto the three-stop ramp with log scaling (RGBA, 0-255)."""
    v = np.log1p(np.asarray(values, dtype=np.float64))
    t = v / v.max() if len(v) and v.max() > 0 else np.zeros_like(v)
    seg = np.minimum((t * 2).astype(int), 1)
    frac = (t * 2 - seg)[:, None]
    rgb = RAMP[seg] + (RAMP[seg + 1] - RAMP[seg]) * frac
    alpha = (95 + 135 * t)[:, None]  # quiet spots let the basemap show through
    return np.hstack([rgb, alpha]).astype(int)


def _photo_grid(samples: pd.DataFrame, limit: int = 12) -> None:
    """Renders thumbnails linking to the original observation/photo, with photographer credit."""
    if samples.empty:
        st.caption("No photos with thumbnails for this selection.")
        return
    cells = []
    for row in samples.head(limit).itertuples():
        credit = row.attribution or (f"© {row.owner_name}" if row.owner_name else "")
        cells.append(
            f'<a href="{html.escape(row.page_url or row.thumbnail_url)}" target="_blank" title="{html.escape(credit)}">'
            f'<img src="{html.escape(row.thumbnail_url)}" loading="lazy" alt="{html.escape(row.title or "photo")}"/>'
            f"<span>{html.escape(row.owner_name or '')}</span></a>"
        )
    st.markdown(f'<div class="photo-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def _selected_hex(event: object) -> int | None:
    """hex_id of the clicked hexagon, from the map selection event or the ?hex= query parameter."""
    try:
        objects = event.selection["objects"].get("hexes") or []  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        objects = []
    if objects:
        st.query_params["hex"] = str(int(objects[0]["hex_id"]))
    value = st.query_params.get("hex")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


# --- data ------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📷 Photo Hotspots")
    data_box = st.expander("Data source", expanded=False)
    with data_box:
        out_dir = st.text_input("Hotspot output folder", value=os.environ.get("HOTSPOTS_DIR", "hotspots"))

if not os.path.exists(os.path.join(out_dir, "meta.json")):
    st.title("Photo Hotspots")
    st.info(
        f"No hotspot outputs found in `{out_dir}`. Create them with:\n\n"
        "```\nspatial-pipeline fetch-inat --bbox denver --output photos.parquet\n"
        "# or: spatial-pipeline ingest-yfcc --input <yfcc metadata> --output photos.parquet --bbox colorado\n"
        "spatial-pipeline hotspots --input photos.parquet --output-dir hotspots\n```\n\n"
        "Then point *Data source* in the sidebar at that folder."
    )
    st.stop()

data = _load(out_dir)
hexes: pd.DataFrame = data["hexes"]  # type: ignore[assignment]
label_counts: pd.DataFrame = data["label_counts"]  # type: ignore[assignment]
local_hotspots: pd.DataFrame = data["local_hotspots"]  # type: ignore[assignment]
timing: pd.DataFrame = data["timing"]  # type: ignore[assignment]
photo_samples: pd.DataFrame = data["photo_samples"]  # type: ignore[assignment]
subjects: pd.DataFrame = data["subjects"]  # type: ignore[assignment]
monthly_counts: pd.DataFrame = data["monthly_counts"]  # type: ignore[assignment]
ranking: pd.DataFrame = data["ranking"]  # type: ignore[assignment]
meta: dict = data["meta"]  # type: ignore[assignment]
alpha = float(meta.get("alpha", 0.05))
resolution = float(meta["resolution"])

# --- controls --------------------------------------------------------------

significant = ranking[ranking["fdr_p_value"] < alpha]
photographers_by_tag = ranking.set_index("label")["photographers"].to_dict()
info = subjects.set_index("label") if not subjects.empty else pd.DataFrame(columns=["group", "scientific_name"])
group_of = info["group"].to_dict()
sci_of = info["scientific_name"].to_dict()
month_rows = timing[timing["kind"] == "month"]


def _name(label: str) -> str:
    sci = sci_of.get(label, "")
    return f"{label} ({sci})" if sci else label


def _pick_subject(label: str) -> None:
    """Button callback: show this subject (clearing a group filter that would hide it)."""
    st.session_state["group"] = "All"
    st.session_state["subject"] = label


with st.sidebar:
    when = st.select_slider(
        "When", options=["All year", *MONTH_NAMES], value="All year", key="when",
        help="Pick a month to see what is in season and where it is photographed that month.",
    )
    month = None if when == "All year" else MONTH_NAMES.index(when) + 1

    present_groups = [g for g in pd.Series([group_of.get(t, "") for t in significant["label"]]).value_counts().index if g]
    group = "All"
    if present_groups:
        group = st.pills("Group", ["All", *present_groups], default="All", key="group") or "All"

    candidates = significant["label"].astype(str)
    if group != "All":
        candidates = candidates[candidates.map(lambda t: group_of.get(t, "") == group)]
    in_month: dict[str, tuple[int, float]] = {}
    if month is not None:
        rows = month_rows[(month_rows["bin"] == month) & month_rows["label"].isin(set(candidates))]
        in_month = {r.label: (int(r.photographers), float(r.lift)) for r in rows.itertuples()}
        candidates = pd.Series(sorted(candidates, key=lambda t: in_month.get(t, (0, 0.0)), reverse=True))

    def _option_text(t: str) -> str:
        if t == ALL_PHOTOS:
            return "Everything"
        if month is not None and t in in_month:
            n, lift = in_month[t]
            return f"{_name(t)}  ·  {n} in {when}" + (f"  ↑{lift:.1f}×" if lift >= 1.5 else "")
        return f"{_name(t)}  ·  {photographers_by_tag.get(t, 0):,} photographers"

    subject_options = [ALL_PHOTOS, *candidates.tolist()]
    if st.session_state.get("subject") not in subject_options:
        st.session_state["subject"] = ALL_PHOTOS
    tag = st.selectbox(
        "What do you want to photograph?",
        subject_options,
        format_func=_option_text,
        key="subject",
        help="Subjects that cluster in specific places (Moran's I, FDR-corrected). "
        "With a month picked, the most photographed that month come first; ↑ marks unusually seasonal ones.",
    )
    only_hot = st.checkbox(
        "Only significant hot spots",
        help=f"Hexes whose neighborhood has significantly more of this subject than expected (Getis-Ord Gi*, FDR < {alpha:g}).",
    )
    min_photographers = st.slider("Minimum photographers per hex", 1, 20, 1)
    st.caption(
        f"{len(significant):,} of {len(ranking):,} subjects cluster in specific places. "
        f"Hexes are {resolution / 1000:g} km across; counts are unique photographers."
    )

# --- per-tag view ----------------------------------------------------------

use_month = month is not None and not monthly_counts.empty
if use_month:
    # Photographers in the chosen month only; hot-spot markers stay all-year (Gi* is computed per year).
    per_hex = monthly_counts[(monthly_counts["label"] == tag) & (monthly_counts["month"] == month)]
    view = hexes.drop(columns="photographers").merge(per_hex[["hex_id", "photographers"]], on="hex_id")
elif tag == ALL_PHOTOS:
    view = hexes.copy()
else:
    per_tag = label_counts[label_counts["label"] == tag][["hex_id", "photographers"]]
    view = hexes.drop(columns="photographers").merge(per_tag, on="hex_id")
view = view.assign(value=view["photographers"])
if tag == ALL_PHOTOS:
    hot = (view["density_gi_z"] > 0) & (view["density_fdr"] < alpha)
    view = view.assign(gi_z=view["density_gi_z"].where(hot))
else:
    tag_hot = local_hotspots.loc[local_hotspots["label"] == tag, ["hex_id", "gi_z"]]
    view = view.merge(tag_hot, on="hex_id", how="left")
hot_count = int(view["gi_z"].notna().sum())
view = view[view["value"] >= min_photographers]
if only_hot:
    view = view[view["gi_z"].notna()]
view = view.reset_index(drop=True)

subject = "everything" if tag == ALL_PHOTOS else tag
month_name = "" if month is None else ["January", "February", "March", "April", "May", "June", "July",
                                       "August", "September", "October", "November", "December"][month - 1]
st.title("Where to shoot " + ("in the area" if tag == ALL_PHOTOS else tag) + (f" in {month_name}" if month_name else ""))
st.caption(
    "Hot spots from geotagged photos, counted by unique photographers. "
    "Click a hexagon for that spot's subjects, photos and best times."
    + (" Counts are for the chosen month; ★ hot spots are based on the whole year." if use_month else "")
)
if month is not None and monthly_counts.empty:
    st.info("These outputs have no per-month counts yet; rerun `spatial-pipeline hotspots` to filter the map by month.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Photos", f"{meta['num_photos']:,}")
c2.metric("Photographers", f"{int(meta.get('num_photographers', hexes['photographers'].sum())):,}")
c3.metric("Spots on the map", f"{len(view):,}")
c4.metric("Significant hot spots", f"{hot_count:,}")

if view.empty:
    st.warning(
        "No hexes match the current filters. Try another month, lower the minimum photographers, "
        "or turn off the hot-spot filter."
    )
    st.stop()

view["polygon"] = _polygons(view["x"].to_numpy(), view["y"].to_numpy(), resolution, str(meta["crs"]))
view[["r_", "g_", "b_", "a_"]] = _colors(view["value"].to_numpy())
view["hot_text"] = np.where(view["gi_z"].notna(), "★ significant hot spot", "")

layer = pdk.Layer(
    "PolygonLayer",
    id="hexes",
    data=view[["hex_id", "polygon", "value", "photos", "hot_text", "r_", "g_", "b_", "a_"]],
    get_polygon="polygon",
    get_fill_color="[r_, g_, b_, a_]",
    get_line_color=[16, 21, 19, 160],
    line_width_min_pixels=0.6,
    pickable=True,
    auto_highlight=True,
    highlight_color=[255, 255, 255, 90],
)
# Hex-center dots with a minimum on-screen size keep small hexes visible when zoomed out.
dots = pdk.Layer(
    "ScatterplotLayer",
    id="dots",
    data=view[["lon", "lat", "r_", "g_", "b_", "a_"]],
    get_position="[lon, lat]",
    get_radius=resolution / 2.0,
    radius_min_pixels=2.5,
    get_fill_color="[r_, g_, b_, a_]",
)
# Fit the view to the central 95% of hexes (Web Mercator: 256 px spans 360 degrees at zoom 0).
lon_lo, lon_hi = view["lon"].quantile([0.025, 0.975])
lat_lo, lat_hi = view["lat"].quantile([0.025, 0.975])
lon_span = max(lon_hi - lon_lo, (lat_hi - lat_lo) / np.cos(np.deg2rad((lat_lo + lat_hi) / 2)), 0.02)
view_state = pdk.ViewState(
    longitude=float((lon_lo + lon_hi) / 2),
    latitude=float((lat_lo + lat_hi) / 2),
    zoom=float(np.clip(np.log2(560 * 360 / (256 * lon_span)) - 0.3, 3, 13)),
)
tooltip = {
    "html": "<b>{value} photographers</b><br/>{photos} photos in this spot<br/><span style='color:#E0A458'>{hot_text}</span>",
    "style": {"backgroundColor": "#1A2220", "color": "#E9E4D8", "fontSize": "12px", "borderRadius": "6px"},
}

layers = [dots, layer]
selected_param = st.query_params.get("hex")
if selected_param is not None and selected_param.isdigit():
    chosen = hexes[hexes["hex_id"] == int(selected_param)]
    if not chosen.empty:
        outline = _polygons(chosen["x"].to_numpy(), chosen["y"].to_numpy(), resolution, str(meta["crs"]))
        layers.append(pdk.Layer(
            "PolygonLayer", id="selected", data=pd.DataFrame({"polygon": outline}), get_polygon="polygon",
            filled=False, stroked=True, get_line_color=[255, 255, 255, 235], line_width_min_pixels=2.5,
        ))

map_col, side_col = st.columns([2.1, 1], gap="large")
with map_col:
    event = st.pydeck_chart(
        # No map_style: Streamlit picks the Carto basemap that matches the app theme (dark here).
        pdk.Deck(layers=layers, initial_view_state=view_state, map_style=None, tooltip=tooltip),  # type: ignore[arg-type]
        height=560,
        on_select="rerun",
        selection_mode="single-object",
        key="map",
    )
    lo, hi = int(view["value"].min()), int(view["value"].max())
    st.markdown(
        f'<div class="legend"><span>{lo}</span><div class="bar"></div><span>{hi:,} photographers per hex '
        "(log scale)</span></div>",
        unsafe_allow_html=True,
    )

selected = _selected_hex(event)
spot = hexes[hexes["hex_id"] == selected] if selected is not None else hexes.iloc[0:0]

with side_col:
    if not spot.empty:
        s = spot.iloc[0]
        if use_month:
            here = monthly_counts[(monthly_counts["hex_id"] == selected) & (monthly_counts["month"] == month)
                                  & (monthly_counts["label"] != ALL_PHOTOS)]
        else:
            here = label_counts[label_counts["hex_id"] == selected]
        here = here.sort_values("photographers", ascending=False)
        hot_here = set(local_hotspots.loc[local_hotspots["hex_id"] == selected, "label"])
        is_hot = (tag == ALL_PHOTOS and s["density_fdr"] < alpha and s["density_gi_z"] > 0) or tag in hot_here
        st.subheader("Selected spot")
        badge = '<span class="badge">★ significant hot spot</span><br/>' if is_hot else ""
        st.markdown(
            f'<div class="spot-facts">{badge}<b>{int(s["photographers"]):,}</b> photographers · '
            f'<b>{int(s["photos"]):,}</b> photos<br/>{s["lat"]:.4f}, {s["lon"]:.4f} · '
            f'<a href="https://www.google.com/maps/dir/?api=1&destination={s["lat"]:.5f},{s["lon"]:.5f}" '
            'target="_blank">Directions ↗</a></div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear selection"):
            st.query_params.pop("hex", None)
            st.rerun()
        is_group = here["label"].isin(GROUP_TAGS)
        groups = here[is_group]
        if not groups.empty:
            st.markdown(
                "**Here:** " + " · ".join(f"{r.label} {r.photographers}" for r in groups.head(6).itertuples())
            )
        st.markdown("**Top subjects here**" + (f" in {month_name}" if use_month else ""))
        top_here = here[~is_group].head(8)
        top_here = top_here.assign(
            subject=np.where(top_here["label"].isin(hot_here), "★ ", "") + top_here["label"].map(_name)
        )
        st.dataframe(
            top_here[["subject", "photographers"]], hide_index=True, use_container_width=True,
            column_config={"subject": st.column_config.TextColumn(help="★ = significant hot spot for this subject")},
        )
        pool = photo_samples[(photo_samples["hex_id"] == selected) & (photo_samples["label"] == tag)]
        if pool.empty:
            pool = photo_samples[(photo_samples["hex_id"] == selected) & (photo_samples["label"] == ALL_PHOTOS)]
        st.markdown("**Photos from this spot**")
        _photo_grid(pool)
    else:
        st.subheader("Everything" if tag == ALL_PHOTOS else _name(tag)[:1].upper() + _name(tag)[1:])
        if tag != ALL_PHOTOS and group_of.get(tag):
            st.caption(group_of[tag].capitalize())
        peaks = describe_peaks(timing, tag)
        if peaks:
            st.markdown(f"**Best times:** {peaks}")
        if month is not None and tag == ALL_PHOTOS:
            # What is photographed unusually often this month (lift), among place-specific subjects.
            season = month_rows[(month_rows["bin"] == month) & month_rows["label"].isin(set(significant["label"]))
                                & (month_rows["photographers"] >= 3) & (month_rows["lift"] >= 1.2)]
            season = season.sort_values(["lift", "photographers"], ascending=False).head(8)
            st.markdown(f"**In season in {month_name}**")
            if season.empty:
                st.caption("Nothing stands out this month yet.")
            for r in season.itertuples():
                st.button(
                    f"{_name(r.label)}  ·  ↑{r.lift:.1f}×", key=f"season-{r.label}",
                    on_click=_pick_subject, args=(r.label,), use_container_width=True,
                    help=f"{int(r.photographers)} photographers in {month_name}",
                )
        top_ids = view.nlargest(6, "value")["hex_id"]
        pool = photo_samples[photo_samples["hex_id"].isin(top_ids) & (photo_samples["label"] == tag)]
        st.markdown("**Photos from the top spots**")
        _photo_grid(pool.drop_duplicates("photo_id"))
        st.caption("👆 Click a hexagon on the map to see that spot.")

# --- when to go ------------------------------------------------------------

tag_timing = timing[timing["label"] == tag]
if not tag_timing.empty:
    st.subheader("When to go")
    peaks = describe_peaks(timing, tag)
    st.caption(
        f"Peak months and hours for **{subject}**: {peaks or 'not enough data'}. "
        "Bars count unique photographers. Compare with *Everything* to see what is truly seasonal."
    )
    months = tag_timing[tag_timing["kind"] == "month"].set_index("bin")
    hours = tag_timing[tag_timing["kind"] == "hour"].set_index("bin")
    m_col, h_col = st.columns(2)
    with m_col:
        st.bar_chart(
            pd.DataFrame({"photographers": months["photographers"].to_numpy()},
                         index=[f"{b:02d} {MONTH_NAMES[b - 1]}" for b in months.index]),
            height=220, color=ACCENT, x_label="month", y_label="photographers",
        )
    with h_col:
        if hours["photographers"].sum() > 0:
            st.bar_chart(
                pd.DataFrame({"photographers": hours["photographers"].to_numpy()},
                             index=[f"{b:02d}:00" for b in hours.index]),
                height=220, color=ACCENT, x_label="hour (local)", y_label="photographers",
            )
        else:
            st.caption("No time-of-day information for this subject.")

# --- tables ----------------------------------------------------------------

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Top spots")
    top = view.nlargest(15, "value")[["value", "photos", "gi_z", "lat", "lon"]].rename(columns={"value": "photographers"})
    top["map"] = [f"https://www.google.com/maps?q={la:.5f},{lo:.5f}" for la, lo in zip(top["lat"], top["lon"])]
    st.dataframe(
        top, hide_index=True, use_container_width=True,
        column_config={
            "gi_z": st.column_config.NumberColumn("hot spot z", format="%.1f", help="Gi* z-score; blank = not significant"),
            "lat": st.column_config.NumberColumn(format="%.4f"),
            "lon": st.column_config.NumberColumn(format="%.4f"),
            "map": st.column_config.LinkColumn("map", display_text="open ↗"),
        },
    )
with right:
    st.subheader("Most place-specific subjects")
    cols = [c for c in ("label", "morans_i", "photographers", "hotspot_hexes", "peak_months") if c in ranking.columns]
    st.dataframe(
        ranking.head(50)[cols].rename(columns={"label": "subject"}), hide_index=True, use_container_width=True,
        column_config={
            "morans_i": st.column_config.ProgressColumn(
                "clustering", min_value=0.0, max_value=float(max(ranking["morans_i"].max(), 1e-9)), format="%.2f",
                help="Moran's I: how strongly the subject concentrates in specific places",
            ),
            "hotspot_hexes": st.column_config.NumberColumn("hot spots"),
            "peak_months": st.column_config.TextColumn("peak months"),
        },
    )
