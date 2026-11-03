"""
Command-line interface (CLI) for Spatial Pipeline powered by Typer and Rich.

Provides commands for ingesting geotagged photo metadata, computing photo hotspots, running the
generic labeled-point pipeline, ranking spatially variable labels, exporting to AnnData / Zarr,
launching the interactive Streamlit map, and benchmarking.
"""

from __future__ import annotations

import os
import time

import typer
from rich.console import Console
from rich.table import Table

from spatial_pipeline.binning import benchmark_binning_algorithms, compute_square_bins
from spatial_pipeline.geo import parse_bbox
from spatial_pipeline.graphs import (
    build_spatial_graph,
    compute_spatially_variable_labels,
)
from spatial_pipeline.hotspots import compute_hotspots, save_hotspots
from spatial_pipeline.io import (
    generate_vectorized_mock_data,
    load_transcript_data,
    save_anndata,
    to_anndata,
    to_zarr,
)
from spatial_pipeline.out_of_core import (
    dask_bin_transcripts,
    profile_out_of_core_execution,
)
from spatial_pipeline.photos import load_photos, read_yfcc_metadata, save_photos

app = typer.Typer(
    help="Spatial Pipeline CLI: spatial hotspot engine for geotagged photos and other labeled points."
)
console = Console()


@app.command("ingest-yfcc")
def ingest_yfcc(
    input: str = typer.Option(..., "--input", "-i", help="YFCC100M metadata: TSV, .bz2/.gz TSV, or SQLite database"),
    output: str = typer.Option("photos.parquet", "--output", "-o", help="Output photo table (.parquet or .csv)"),
    bbox: str = typer.Option("colorado", "--bbox", "-b", help="Region name (colorado, denver) or 'min_lon,min_lat,max_lon,max_lat'"),
    min_accuracy: int = typer.Option(12, "--min-accuracy", help="Minimum Flickr geotag accuracy (1 = world ... 16 = street); 0 disables"),
    include_videos: bool = typer.Option(False, "--include-videos", help="Keep videos as well as photos"),
    chunksize: int = typer.Option(500_000, "--chunksize", help="Raw rows read per chunk"),
) -> None:
    """
    Stream YFCC100M metadata, keep geotagged photos inside a region, and write a clean photo table.
    """
    region = parse_bbox(bbox)
    console.print(f"[bold blue][*] Reading {input} (bbox={region}, min_accuracy={min_accuracy})...[/bold blue]")
    t0 = time.perf_counter()
    photos = read_yfcc_metadata(input, bbox=region, min_accuracy=min_accuracy, include_videos=include_videos, chunksize=chunksize)
    elapsed = time.perf_counter() - t0
    save_photos(photos, output)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Photos kept", f"{len(photos):,}")
    table.add_row("Photographers", f"{photos['user_id'].nunique():,}")
    table.add_row("Photos with tags", f"{(photos['tags'] != '').sum():,}")
    table.add_row("Elapsed", f"{elapsed:.1f} s")
    table.add_row("Output", output)
    console.print(table)


@app.command()
def hotspots(
    input: str = typer.Option("photos.parquet", "--input", "-i", help="Photo table from `ingest-yfcc`"),
    output_dir: str = typer.Option("hotspots", "--output-dir", "-o", help="Directory for hotspot outputs"),
    resolution: float = typer.Option(1000.0, "--resolution", "-r", help="Hex center-to-center distance in meters"),
    max_labels: int = typer.Option(300, "--max-labels", help="Analyze at most this many tags"),
    min_label_photographers: int = typer.Option(10, "--min-label-photographers", help="Minimum distinct photographers per tag"),
    min_hex_photographers: int = typer.Option(3, "--min-hex-photographers", help="Minimum distinct photographers for a hex to enter the statistics"),
    neighbor_rings: int = typer.Option(1, "--neighbor-rings", help="Hexes within this many rings are neighbors (1 = touching)"),
    normalize: str = typer.Option("share", "--normalize", help="'share' (tag fraction per hex) or 'count' (raw photographer counts)"),
    top_n: int = typer.Option(20, "--top-n", "-n", help="Number of top clustered tags to print"),
) -> None:
    """
    Bin photos into hexes, count unique photographers, and rank tags by spatial clustering (Moran's I).
    """
    if normalize not in ("share", "count"):
        raise typer.BadParameter("--normalize must be 'share' or 'count'")

    photos = load_photos(input)
    console.print(f"[bold blue][*] Computing hotspots for {len(photos):,} photos at {resolution:g} m hexes...[/bold blue]")
    result = compute_hotspots(
        photos,
        resolution=resolution,
        max_labels=max_labels,
        min_label_photographers=min_label_photographers,
        min_hex_photographers=min_hex_photographers,
        neighbor_rings=neighbor_rings,
        normalize=normalize,  # type: ignore[arg-type]
    )
    save_hotspots(result, output_dir, extra_meta={"input": input, "normalize": normalize})

    console.print(
        f"[bold green][+] {len(result.hexes):,} active hexes ({len(result.stats_hex_ids):,} used for statistics), "
        f"{len(result.labels):,} tags analyzed.[/bold green]"
    )
    table = Table(title=f"Top {top_n} Spatially Clustered Tags", show_header=True, header_style="bold green")
    table.add_column("Rank", style="dim")
    table.add_column("Tag", style="bold cyan")
    table.add_column("Moran's I")
    table.add_column("FDR p-val")
    table.add_column("Photographers")
    table.add_column("Hexes")
    for rank, row in enumerate(result.ranking.head(top_n).itertuples(), start=1):
        table.add_row(
            str(rank),
            str(row.label),
            f"{row.morans_i:.4f}",
            f"{row.fdr_p_value:.2e}",
            f"{row.photographers:,}",
            f"{row.hexes:,}",
        )
    console.print(table)
    console.print(f"[bold green][+] Wrote outputs to {output_dir}/ (open the map with `spatial-pipeline ui`)[/bold green]")


@app.command()
def run(
    input: str | None = typer.Option(None, "--input", "-i", help="Path to input spatial transcripts CSV"),
    resolution: float = typer.Option(10.0, "--resolution", "-r", help="Spatial grid resolution size"),
    max_dist: float = typer.Option(15.0, "--max-dist", "-m", help="Maximum radius distance for adjacency graph"),
    backend: str = typer.Option("auto", "--backend", "-b", help="Graph backend (auto, cuml, cugraph, multiprocessing, threading, cpu)"),
    n_jobs: int = typer.Option(1, "--n-jobs", "-j", help="Number of parallel worker processes (-1 for all cores)"),
    use_arrow: bool = typer.Option(False, "--use-arrow", "-a", help="Use zero-copy PyArrow engine for loading data"),
    use_dask: bool = typer.Option(False, "--use-dask", "-d", help="Use out-of-core Dask DataFrame spatial binning"),
    generate_mock: bool = typer.Option(False, "--generate-mock", help="Generate mock transcripts CSV before running")
) -> None:
    """
    Run spatial transcriptomics processing pipeline.
    """
    input_file = input
    if not input_file or generate_mock:
        input_file = "mock_transcripts.csv"
        console.print("[bold yellow][!] No input specified or mock flag set. Generating mock data...[/bold yellow]")
        generate_vectorized_mock_data(input_file)

    if use_dask:
        console.print(f"[bold blue][*] Running Dask out-of-core distributed binning on {input_file}...[/bold blue]")
        sparse_counts, bin_coords, _ = dask_bin_transcripts(input_file, resolution=resolution)
        total_molecules = -1
    else:
        console.print(f"[bold blue][*] Loading dataset: {input_file} (use_arrow={use_arrow})...[/bold blue]")
        df = load_transcript_data(input_file, use_arrow=use_arrow)
        total_molecules = len(df)
        console.print(f"[bold green][+] Loaded {total_molecules:,} total transcript molecules.[/bold green]")

        console.print(f"[bold blue][*] Binning spatial grid (Resolution: {resolution})...[/bold blue]")
        sparse_counts, bin_coords = compute_square_bins(df, resolution)

    console.print(f"[bold green][+] Created {sparse_counts.shape[0]} unique active bins across {sparse_counts.shape[1]} labels.[/bold green]")

    console.print(f"[bold blue][*] Building spatial adjacency graph (Max Dist: {max_dist}, Backend: {backend}, n_jobs: {n_jobs})...[/bold blue]")
    adj, laplacian = build_spatial_graph(bin_coords, max_dist, n_jobs=n_jobs, backend=backend)  # type: ignore

    console.print("\n[bold green]=== Execution Summary ===[/bold green]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    if total_molecules > 0:
        table.add_row("Total Transcripts", f"{total_molecules:,}")
    table.add_row("Unique Active Bins", str(sparse_counts.shape[0]))
    table.add_row("Unique Labels", str(sparse_counts.shape[1]))
    table.add_row("Graph Non-Zero Edges", f"{adj.nnz:,}")
    table.add_row("Laplacian Matrix Shape", str(laplacian.shape))
    console.print(table)


@app.command()
def svg(
    input: str | None = typer.Option(None, "--input", "-i", help="Path to transcript CSV file"),
    resolution: float = typer.Option(10.0, "--resolution", "-r", help="Spatial grid resolution size"),
    max_dist: float = typer.Option(15.0, "--max-dist", "-m", help="Maximum radius distance for adjacency graph"),
    top_n: int = typer.Option(10, "--top-n", "-n", help="Number of top spatially variable labels to display"),
    output: str | None = typer.Option(None, "--output", "-o", help="Optional CSV output file path for SVG results")
) -> None:
    """
    Compute Spatially Variable Labels (SVGs) via Moran's I & Geary's C spatial autocorrelation.
    """
    input_file = input
    if not input_file:
        input_file = "mock_transcripts.csv"
        if not os.path.exists(input_file):
            generate_vectorized_mock_data(input_file)

    console.print(f"[bold blue][*] Loading dataset for SVG calculation: {input_file}...[/bold blue]")
    df = load_transcript_data(input_file)
    sparse_counts, bin_coords = compute_square_bins(df, resolution)
    label_list = df["label"].cat.categories.to_numpy()

    adj, _ = build_spatial_graph(bin_coords, max_dist)
    svg_df = compute_spatially_variable_labels(sparse_counts, label_list, adj, top_n=top_n)

    table = Table(title=f"Top {top_n} Spatially Variable Labels (Moran's I)", show_header=True, header_style="bold green")
    table.add_column("Rank", style="dim")
    table.add_column("Label", style="bold cyan")
    table.add_column("Moran's I")
    table.add_column("Geary's C")
    table.add_column("z-Score")
    table.add_column("FDR p-val")

    for rank, row in enumerate(svg_df.itertuples(), start=1):
        table.add_row(
            str(rank),
            str(row.label),
            f"{row.morans_i:.4f}",
            f"{row.gearys_c:.4f}",
            f"{row.z_score:.2f}",
            f"{row.fdr_p_value:.2e}"
        )
    console.print(table)

    if output:
        svg_df.to_csv(output, index=False)
        console.print(f"[bold green][+] Exported SVG table to: {output}[/bold green]")


@app.command()
def export(
    input: str = typer.Option(..., "--input", "-i", help="Path to input transcript CSV file"),
    output: str = typer.Option("spatial_processed.h5ad", "--output", "-o", help="Output file path (.h5ad or .zarr)"),
    resolution: float = typer.Option(10.0, "--resolution", "-r", help="Spatial grid resolution size"),
    max_dist: float = typer.Option(15.0, "--max-dist", "-m", help="Maximum radius distance for adjacency graph"),
    format: str = typer.Option("h5ad", "--format", "-f", help="Output format: h5ad or zarr"),
    compute_svg: bool = typer.Option(True, "--svg/--no-svg", help="Compute Moran's I & Geary's C into AnnData .var")
) -> None:
    """
    Process transcripts CSV and export directly to AnnData (.h5ad) or Zarr format.
    """
    console.print(f"[bold blue][*] Loading dataset: {input}...[/bold blue]")
    df = load_transcript_data(input)
    sparse_counts, bin_coords = compute_square_bins(df, resolution)
    label_list = df["label"].cat.categories.to_numpy()

    console.print("[bold blue][*] Building spatial adjacency graph...[/bold blue]")
    adj, laplacian = build_spatial_graph(bin_coords, max_dist)

    console.print("[bold blue][*] Converting to AnnData object...[/bold blue]")
    adata = to_anndata(
        sparse_counts=sparse_counts,
        bin_coords=bin_coords,
        labels=label_list,
        adjacency_matrix=adj,
        laplacian_matrix=laplacian,
        compute_svg=compute_svg
    )

    if format.lower() == "zarr" or output.endswith(".zarr"):
        to_zarr(adata, output)
        console.print(f"[bold green][+] Successfully exported chunked Zarr hierarchy to: {output}[/bold green]")
    else:
        save_anndata(adata, output)
        console.print(f"[bold green][+] Successfully exported AnnData to: {output}[/bold green]")


@app.command()
def ui() -> None:
    """
    Launch the Streamlit photo hotspot map.
    """
    import subprocess
    import sys
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app.py")
    if not os.path.exists(app_path):
        app_path = "app.py"
    console.print(f"[bold green][*] Launching Streamlit dashboard from {app_path}...[/bold green]")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=False)


@app.command()
def benchmark(
    num_molecules: int = typer.Option(20000, "--num-molecules", "-n", help="Number of synthetic molecules for benchmarking"),
    resolution: float = typer.Option(10.0, "--resolution", "-r", help="Spatial grid resolution"),
    tile_size: float = typer.Option(50.0, "--tile-size", "-t", help="Tile size for out-of-core benchmark"),
    max_dist: float = typer.Option(15.0, "--max-dist", "-m", help="Max graph radius distance")
) -> None:
    """
    Benchmark spatial binning algorithms (Spatial Hashing vs KD-Tree vs Quadtree) and profile out-of-core peak memory usage.
    """
    mock_file = "tmp_benchmark_data.csv"
    console.print(f"[bold blue][*] Generating {num_molecules} synthetic transcripts for benchmarking...[/bold blue]")
    generate_vectorized_mock_data(mock_file, num_molecules=num_molecules)

    df = load_transcript_data(mock_file)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    console.print("\n[bold cyan]1. Benchmarking Spatial Binning Algorithms...[/bold cyan]")
    bench_df = benchmark_binning_algorithms(x, y, resolution)

    table = Table(title="Binning Algorithm Tradeoffs", show_header=True, header_style="bold cyan")
    for col in bench_df.columns:
        table.add_column(col)
    for _, row in bench_df.iterrows():
        table.add_row(*[str(val) for val in row.values])
    console.print(table)

    console.print("\n[bold cyan]2. Profiling Out-of-Core Processing Memory & Runtime...[/bold cyan]")
    ooc_report = profile_out_of_core_execution(mock_file, tile_size, resolution, max_dist)

    mem_table = Table(title="Out-of-Core Stage Profiling (Peak RAM)", show_header=True, header_style="bold yellow")
    mem_table.add_column("Pipeline Stage")
    mem_table.add_column("Peak Memory (MB)")
    mem_table.add_column("Elapsed Time (sec)")

    for stage in ["partitioning", "tile_processing", "stitching"]:
        s_data = ooc_report[stage]
        mem_table.add_row(stage.capitalize(), f"{s_data['peak_mb']:.4f}", f"{s_data['elapsed_sec']:.4f}")
    console.print(mem_table)

    summary = ooc_report["summary"]
    console.print(f"[bold green][+] Total Out-of-Core Time: {summary['total_elapsed_sec']:.3f} sec | Peak Memory: {summary['max_stage_peak_mb']:.4f} MB[/bold green]")

    if os.path.exists(mock_file):
        os.remove(mock_file)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
