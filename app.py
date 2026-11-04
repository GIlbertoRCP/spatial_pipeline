"""
Streamlit map of photo hotspots produced by `spatial-pipeline hotspots`.

Run with `spatial-pipeline ui` or `streamlit run app.py`.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from spatial_pipeline.hotspots import hex_polygons_lonlat, load_hotspots

st.set_page_config(page_title="Photo Hotspots", layout="wide")

ALL_PHOTOS = "(all photos)"


@st.cache_data
def _load(out_dir: str) -> dict[str, object]:
    return load_hotspots(out_dir)


@st.cache_data
def _polygons(x: np.ndarray, y: np.ndarray, resolution: float, crs: str) -> list[list[list[float]]]:
    return hex_polygons_lonlat(x, y, resolution, crs)


def _colors(values: np.ndarray) -> np.ndarray:
    """Maps values to a light-yellow -> deep-red ramp with log scaling (RGBA, 0-255)."""
    v = np.log1p(np.asarray(values, dtype=np.float64))
    t = v / v.max() if len(v) and v.max() > 0 else np.zeros_like(v)
    low = np.array([255, 237, 160])
    high = np.array([189, 0, 38])
    rgb = low[None, :] + (high - low)[None, :] * t[:, None]
    alpha = np.full((len(t), 1), 190)
    return np.hstack([rgb, alpha]).astype(int)


st.title("Photo Hotspots")
st.caption("Where people photograph, counted by unique photographers per hex. Tags ranked by spatial clustering (Moran's I).")

with st.sidebar:
    out_dir = st.text_input("Hotspot output directory", value=os.environ.get("HOTSPOTS_DIR", "hotspots"))

if not os.path.exists(os.path.join(out_dir, "meta.json")):
    st.info(
        f"No hotspot outputs found in `{out_dir}`. Create them with:\n\n"
        "```\nspatial-pipeline fetch-inat --bbox denver --output photos.parquet\n"
        "# or: spatial-pipeline ingest-yfcc --input <yfcc metadata> --output photos.parquet --bbox colorado\n"
        "spatial-pipeline hotspots --input photos.parquet --output-dir hotspots\n```"
    )
    st.stop()

data = _load(out_dir)
hexes: pd.DataFrame = data["hexes"]  # type: ignore[assignment]
label_counts: pd.DataFrame = data["label_counts"]  # type: ignore[assignment]
local_hotspots: pd.DataFrame = data["local_hotspots"]  # type: ignore[assignment]
ranking: pd.DataFrame = data["ranking"]  # type: ignore[assignment]
meta: dict = data["meta"]  # type: ignore[assignment]

with st.sidebar:
    significant = ranking[ranking["fdr_p_value"] < 0.05]
    options = [ALL_PHOTOS] + significant["label"].astype(str).tolist()
    tag = st.selectbox("Tag (significantly clustered, strongest first)", options)
    min_photographers = st.slider("Minimum photographers per hex", 1, 20, 1)
    alpha = float(meta.get("alpha", 0.05))
    only_hot = st.checkbox(
        "Only significant hot spots",
        help=f"Hexes whose neighborhood is significantly above average (Getis-Ord Gi*, FDR < {alpha:g}).",
    )
    st.caption(f"{len(significant):,} of {len(ranking):,} analyzed tags are significantly clustered (FDR < 0.05).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Photos", f"{meta['num_photos']:,}")
c2.metric("Photographers (sum over hexes)", f"{int(hexes['photographers'].sum()):,}")
c3.metric("Active hexes", f"{meta['num_hexes']:,}")
c4.metric("Hex size", f"{meta['resolution'] / 1000:g} km")

if tag == ALL_PHOTOS:
    view = hexes.assign(value=hexes["photographers"])
    hot = (view["density_gi_z"] > 0) & (view["density_fdr"] < alpha)
    view = view.assign(gi_z=view["density_gi_z"].where(hot))
else:
    per_tag = label_counts[label_counts["label"] == tag][["hex_id", "photographers"]]
    view = hexes.drop(columns="photographers").merge(per_tag, on="hex_id")
    view = view.assign(value=view["photographers"])
    tag_hot = local_hotspots.loc[local_hotspots["label"] == tag, ["hex_id", "gi_z"]]
    view = view.merge(tag_hot, on="hex_id", how="left")
view = view[view["value"] >= min_photographers]
if only_hot:
    view = view[view["gi_z"].notna()]
view = view.reset_index(drop=True)
view["hotspot"] = np.where(
    view["gi_z"].notna(), "hot spot, Gi* z = " + view["gi_z"].round(1).astype(str), "not significant"
)

if view.empty:
    st.warning("No hexes match the current filters.")
    st.stop()

view["polygon"] = _polygons(view["x"].to_numpy(), view["y"].to_numpy(), float(meta["resolution"]), str(meta["crs"]))
view[["r_", "g_", "b_", "a_"]] = _colors(view["value"].to_numpy())

layer = pdk.Layer(
    "PolygonLayer",
    data=view[["polygon", "value", "photos", "hotspot", "lon", "lat", "r_", "g_", "b_", "a_"]],
    get_polygon="polygon",
    get_fill_color="[r_, g_, b_, a_]",
    get_line_color=[80, 80, 80, 80],
    line_width_min_pixels=0.5,
    pickable=True,
)
# Hex-center dots with a minimum on-screen size keep small hexes visible when zoomed out.
dots = pdk.Layer(
    "ScatterplotLayer",
    data=view[["lon", "lat", "r_", "g_", "b_", "a_"]],
    get_position="[lon, lat]",
    get_radius=float(meta["resolution"]) / 2.0,
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
    zoom=float(np.clip(np.log2(700 * 360 / (256 * lon_span)) - 0.3, 3, 13)),
)
st.pydeck_chart(
    pdk.Deck(
        layers=[dots, layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="light",
        tooltip={"text": "{value} photographers\n{photos} photos in hex\n{hotspot}\n{lat}, {lon}"},
    ),
    height=620,
)

left, right = st.columns(2)
with left:
    st.subheader("Top hexes")
    top = view.nlargest(15, "value")[["value", "photos", "gi_z", "lat", "lon"]].rename(columns={"value": "photographers"})
    top["map"] = [f"https://www.google.com/maps?q={la:.5f},{lo:.5f}" for la, lo in zip(top["lat"], top["lon"])]
    st.dataframe(top, hide_index=True, column_config={"map": st.column_config.LinkColumn("map", display_text="open")})
with right:
    st.subheader("Most clustered tags")
    st.dataframe(
        ranking.head(50)[["label", "morans_i", "fdr_p_value", "photographers", "hexes", "hotspot_hexes"]],
        hide_index=True,
    )
