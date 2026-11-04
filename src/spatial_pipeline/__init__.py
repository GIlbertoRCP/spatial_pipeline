"""
spatial_pipeline: Spatial hotspot engine for geotagged photos and other labeled points.

Provides iNaturalist and YFCC100M photo ingestion, lon/lat projection, hex/square/tree binning,
photographer-weighted hotspot analysis with global (Moran's I / Geary's C) and local (Getis-Ord Gi*)
spatial statistics, CPU/GPU spatial graphs, out-of-core tiling, and PyG export.
"""

from __future__ import annotations

from .binning import (
    benchmark_binning_algorithms,
    bin_hexagonal,
    bin_kdtree,
    bin_quadtree,
    bin_square,
    bin_square_memmap,
    bin_voronoi,
    build_sparse_matrix,
    compute_square_bins,
    get_bin_centers,
)
from .flickr import (
    FlickrClient,
    FlickrError,
    FlickrFetcher,
    photos_from_response,
)
from .geo import (
    COLORADO_BBOX,
    DEFAULT_CRS,
    DENVER_METRO_BBOX,
    in_bbox,
    parse_bbox,
    project_lonlat,
    unproject_xy,
)
from .graphs import (
    build_knn_graph,
    build_spatial_graph,
    compute_gearys_c,
    compute_gpu_knn_graph_cuml,
    compute_laplacian_spectrum,
    compute_morans_i,
    compute_spatially_variable_labels,
    is_gpu_available,
    to_pyg_data,
)
from .hotspots import (
    HotspotResult,
    compute_hotspots,
    hex_polygons_lonlat,
    load_hotspots,
    save_hotspots,
)
from .inat import (
    INatClient,
    INatError,
    INatFetcher,
    observations_to_photos,
)
from .io import load_coordinates
from .out_of_core import (
    dask_bin_points,
    enforce_memory_limit,
    load_tile_with_halo,
    partition_dataset_to_tiles,
    process_tile_out_of_core,
    process_tiles_parallel_out_of_core,
    profile_memory,
    profile_out_of_core_execution,
    stitch_out_of_core_results,
)
from .photos import (
    decode_tags,
    explode_tags,
    load_photos,
    read_yfcc_metadata,
    save_photos,
)

__version__ = "0.5.0"

__all__ = [
    "COLORADO_BBOX",
    "DEFAULT_CRS",
    "DENVER_METRO_BBOX",
    "FlickrClient",
    "FlickrError",
    "FlickrFetcher",
    "HotspotResult",
    "INatClient",
    "INatError",
    "INatFetcher",
    "benchmark_binning_algorithms",
    "bin_hexagonal",
    "bin_kdtree",
    "bin_quadtree",
    "bin_square",
    "bin_square_memmap",
    "bin_voronoi",
    "build_knn_graph",
    "build_sparse_matrix",
    "build_spatial_graph",
    "compute_gearys_c",
    "compute_gpu_knn_graph_cuml",
    "compute_hotspots",
    "compute_laplacian_spectrum",
    "compute_morans_i",
    "compute_spatially_variable_labels",
    "compute_square_bins",
    "dask_bin_points",
    "decode_tags",
    "enforce_memory_limit",
    "explode_tags",
    "get_bin_centers",
    "hex_polygons_lonlat",
    "in_bbox",
    "is_gpu_available",
    "load_coordinates",
    "load_hotspots",
    "load_photos",
    "load_tile_with_halo",
    "observations_to_photos",
    "parse_bbox",
    "partition_dataset_to_tiles",
    "photos_from_response",
    "process_tile_out_of_core",
    "process_tiles_parallel_out_of_core",
    "profile_memory",
    "profile_out_of_core_execution",
    "project_lonlat",
    "read_yfcc_metadata",
    "save_hotspots",
    "save_photos",
    "stitch_out_of_core_results",
    "to_pyg_data",
    "unproject_xy",
]
