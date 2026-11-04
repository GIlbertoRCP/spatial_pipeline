"""
Command-line interface (CLI) for Spatial Pipeline powered by Typer and Rich.

Provides commands for ingesting geotagged photo metadata (iNaturalist, YFCC100M, Flickr), computing
photo hotspots, and launching the interactive Streamlit map.
"""

from __future__ import annotations

import os
import time

import typer
from rich.console import Console
from rich.table import Table

from spatial_pipeline.geo import parse_bbox
from spatial_pipeline.hotspots import (
    compute_hotspots,
    monthly_label_counts,
    sample_photos,
    save_hotspots,
    save_photo_samples,
    subject_info,
)
from spatial_pipeline.photos import load_photos, read_yfcc_metadata, save_photos
from spatial_pipeline.timing import MONTH_NAMES, compute_timing, peak_bins, save_timing

app = typer.Typer(
    help="Spatial Pipeline CLI: find wildlife and nature photo hotspots in geotagged photos."
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


@app.command("fetch-flickr")
def fetch_flickr(
    api_key: str = typer.Option(
        ..., "--api-key", envvar="FLICKR_API_KEY", show_default=False,
        help="Flickr API key (prefer the FLICKR_API_KEY environment variable over the flag)",
    ),
    output: str = typer.Option("photos_flickr.parquet", "--output", "-o", help="Output photo table (.parquet or .csv)"),
    bbox: str = typer.Option("colorado", "--bbox", "-b", help="Region name (colorado, denver) or 'min_lon,min_lat,max_lon,max_lat'"),
    start: str = typer.Option("2015-01-01", "--start", help="Upload date start, inclusive (YYYY-MM-DD, UTC)"),
    end: str = typer.Option(..., "--end", help="Upload date end, exclusive (YYYY-MM-DD, UTC)"),
    cache_dir: str = typer.Option(".flickr_cache", "--cache-dir", help="Per-window cache; rerun with the same directory to resume"),
    min_accuracy: int = typer.Option(12, "--min-accuracy", help="Minimum Flickr geotag accuracy (1 = world ... 16 = street); 0 disables"),
    open_licenses: bool = typer.Option(False, "--open-licenses", help="Only Creative Commons / public-domain photos"),
    min_interval: float = typer.Option(1.0, "--min-interval", help="Seconds between API requests (1.0 = Flickr's 3,600/hour limit)"),
) -> None:
    """
    Fetch geotagged Flickr photos for a region and upload-date range (requires a Flickr Pro API key).
    """
    from datetime import datetime, timezone

    from spatial_pipeline.flickr import (
        OPEN_LICENSES,
        FlickrClient,
        FlickrError,
        FlickrFetcher,
        filter_accuracy,
    )

    region = parse_bbox(bbox)
    try:
        t_start = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        t_end = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise typer.BadParameter("--start and --end must be YYYY-MM-DD") from e
    if t_end <= t_start:
        raise typer.BadParameter("--end must be after --start")

    client = FlickrClient(api_key, min_interval=min_interval)
    fetcher = FlickrFetcher(client, cache_dir, licenses=OPEN_LICENSES if open_licenses else None)
    console.print(f"[bold blue][*] Fetching Flickr photos uploaded {start} .. {end} in bbox={region} (cache: {cache_dir})...[/bold blue]")
    t0 = time.perf_counter()
    try:
        photos = fetcher.fetch(region, t_start, t_end, progress=lambda line: console.print(f"    {line}"))
    except FlickrError as e:
        console.print(f"[bold red][x] {e}. Finished windows are cached in {cache_dir}; rerun to resume.[/bold red]")
        raise typer.Exit(1) from e
    photos = filter_accuracy(photos, min_accuracy)
    elapsed = time.perf_counter() - t0
    save_photos(photos, output)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Photos kept", f"{len(photos):,}")
    table.add_row("Photographers", f"{photos['user_id'].nunique():,}")
    table.add_row("API requests (this run)", f"{client.requests_made:,}")
    table.add_row("Truncated windows", f"{len(fetcher.truncated_windows):,}")
    table.add_row("Elapsed", f"{elapsed:.1f} s")
    table.add_row("Output", output)
    console.print(table)
    if fetcher.truncated_windows:
        console.print("[bold yellow][!] Some windows exceeded Flickr's 4,000-result cap even after splitting; "
                      "their extra photos are missing.[/bold yellow]")


@app.command("fetch-inat")
def fetch_inat(
    output: str = typer.Option("photos_inat.parquet", "--output", "-o", help="Output photo table (.parquet or .csv)"),
    bbox: str = typer.Option("colorado", "--bbox", "-b", help="Region name (colorado, denver) or 'min_lon,min_lat,max_lon,max_lat'"),
    cache_dir: str = typer.Option(".inat_cache", "--cache-dir", help="Per-page cache; rerun with the same directory to resume"),
    quality: str = typer.Option("research", "--quality", help="research (verified IDs), needs_id, casual, or all"),
    all_licenses: bool = typer.Option(False, "--all-licenses", help="Include 'all rights reserved' photos (default: Creative Commons only)"),
    observed_from: str | None = typer.Option(None, "--observed-from", help="First observation date (YYYY-MM-DD)"),
    observed_to: str | None = typer.Option(None, "--observed-to", help="Last observation date (YYYY-MM-DD)"),
    max_accuracy_m: float | None = typer.Option(1000.0, "--max-accuracy-m", help="Drop observations with location uncertainty above this many meters"),
    max_requests: int | None = typer.Option(9000, "--max-requests", help="Stop after this many requests (iNaturalist asks for ~10k/day); rerun to continue"),
    min_interval: float = typer.Option(1.0, "--min-interval", help="Seconds between API requests (iNaturalist asks for about 1/second)"),
) -> None:
    """
    Fetch photographed iNaturalist observations (no API key needed): observers act as photographers, species as tags.
    """
    from spatial_pipeline.inat import (
        INatClient,
        INatError,
        INatFetcher,
        filter_positional_accuracy,
    )

    if quality not in ("research", "needs_id", "casual", "all"):
        raise typer.BadParameter("--quality must be research, needs_id, casual or all")
    region = parse_bbox(bbox)
    client = INatClient(min_interval=min_interval)
    fetcher = INatFetcher(client, cache_dir)
    query = INatFetcher.build_query(
        region,
        quality_grade=None if quality == "all" else quality,
        open_licenses=not all_licenses,
        observed_from=observed_from,
        observed_to=observed_to,
    )
    console.print(f"[bold blue][*] Fetching iNaturalist observations in bbox={region} (quality={quality}, cache: {cache_dir})...[/bold blue]")
    t0 = time.perf_counter()
    try:
        photos = fetcher.fetch(query, max_requests=max_requests, progress=lambda line: console.print(f"    {line}"))
    except INatError as e:
        console.print(f"[bold red][x] {e}[/bold red]")
        raise typer.Exit(1) from e
    photos = filter_positional_accuracy(photos, max_accuracy_m)
    elapsed = time.perf_counter() - t0
    save_photos(photos, output)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Observations kept", f"{len(photos):,}")
    table.add_row("Observers", f"{photos['user_id'].nunique():,}")
    table.add_row("Species / taxa", f"{photos['taxon_id'].nunique():,}")
    table.add_row("API requests (this run)", f"{fetcher.requests_this_run:,}")
    table.add_row("Fetch complete", "yes" if fetcher.complete else "no (rerun to continue)")
    table.add_row("Elapsed", f"{elapsed:.1f} s")
    table.add_row("Output", output)
    console.print(table)
    if not fetcher.complete:
        console.print(f"[bold yellow][!] Stopped after {fetcher.requests_this_run:,} requests. Rerun the same command "
                      "(e.g. tomorrow) to continue; the output then includes everything fetched so far.[/bold yellow]")


@app.command()
def hotspots(
    input: str = typer.Option("photos.parquet", "--input", "-i", help="Photo table from `fetch-inat`, `ingest-yfcc` or `fetch-flickr`"),
    output_dir: str = typer.Option("hotspots", "--output-dir", "-o", help="Directory for hotspot outputs"),
    resolution: float = typer.Option(1000.0, "--resolution", "-r", help="Hex center-to-center distance in meters"),
    max_labels: int = typer.Option(300, "--max-labels", help="Analyze at most this many tags"),
    min_label_photographers: int = typer.Option(10, "--min-label-photographers", help="Minimum distinct photographers per tag"),
    min_hex_photographers: int = typer.Option(3, "--min-hex-photographers", help="Minimum distinct photographers for a hex to enter the statistics"),
    neighbor_rings: int = typer.Option(1, "--neighbor-rings", help="Hexes within this many rings are neighbors (1 = touching)"),
    normalize: str = typer.Option("share", "--normalize", help="'share' (tag fraction per hex) or 'count' (raw photographer counts)"),
    alpha: float = typer.Option(0.05, "--alpha", help="FDR level for local Gi* hot spots"),
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
        alpha=alpha,
    )
    timing = compute_timing(photos, labels=result.labels)
    result.ranking["peak_months"] = [
        ", ".join(MONTH_NAMES[m - 1] for m in peak_bins(timing, label, "month")) for label in result.ranking["label"]
    ]
    save_hotspots(result, output_dir, extra_meta={"input": input, "normalize": normalize, "alpha": alpha})
    save_timing(timing, output_dir)
    save_photo_samples(sample_photos(photos, result), output_dir)
    subject_info(photos, result.labels).to_parquet(os.path.join(output_dir, "subjects.parquet"), index=False)
    monthly_label_counts(photos, result).to_parquet(os.path.join(output_dir, "monthly_counts.parquet"), index=False)

    console.print(
        f"[bold green][+] {len(result.hexes):,} active hexes ({len(result.stats_hex_ids):,} used for statistics), "
        f"{len(result.labels):,} tags analyzed, {len(result.local_hotspots):,} significant tag hot spots "
        f"(Gi*, FDR < {alpha:g}).[/bold green]"
    )
    table = Table(title=f"Top {top_n} Spatially Clustered Tags", show_header=True, header_style="bold green")
    table.add_column("Rank", style="dim")
    table.add_column("Tag", style="bold cyan")
    table.add_column("Moran's I")
    table.add_column("FDR p-val")
    table.add_column("Photographers")
    table.add_column("Hexes")
    table.add_column("Hot-spot hexes")
    table.add_column("Peak months")
    for rank, row in enumerate(result.ranking.head(top_n).itertuples(), start=1):
        table.add_row(
            str(rank),
            str(row.label),
            f"{row.morans_i:.4f}",
            f"{row.fdr_p_value:.2e}",
            f"{row.photographers:,}",
            f"{row.hexes:,}",
            f"{row.hotspot_hexes:,}",
            str(row.peak_months),
        )
    console.print(table)
    console.print(f"[bold green][+] Wrote outputs to {output_dir}/ (open the map with `spatial-pipeline ui`)[/bold green]")


@app.command()
def ui() -> None:
    """
    Launch the Streamlit photo hotspot map.
    """
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None or importlib.util.find_spec("pydeck") is None:
        console.print("[bold red][x] The map needs the UI extras: pip install 'spatial_pipeline[ui]'[/bold red]")
        raise typer.Exit(1)
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app.py")
    if not os.path.exists(app_path):
        app_path = "app.py"
    console.print(f"[bold green][*] Launching Streamlit dashboard from {app_path}...[/bold green]")
    # Run from the app folder so .streamlit/config.toml (the theme) is picked up.
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=False, cwd=os.path.dirname(os.path.abspath(app_path)))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
