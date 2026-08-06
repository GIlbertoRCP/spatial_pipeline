import os
import typer
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

from spatial_pipeline.io import load_transcript_data, generate_vectorized_mock_data
from spatial_pipeline.binning import compute_square_bins, benchmark_binning_algorithms
from spatial_pipeline.graphs import build_spatial_graph
from spatial_pipeline.out_of_core import profile_out_of_core_execution

app = typer.Typer(
    help="Spatial Pipeline CLI: High-performance spatial transcriptomics binning, graph generation, & benchmarking."
)
console = Console()

@app.command()
def run(
    input: str = typer.Option(None, "--input", "-i", help="Path to input spatial transcripts CSV"),
    resolution: float = typer.Option(10.0, "--resolution", "-r", help="Spatial grid resolution size"),
    max_dist: float = typer.Option(15.0, "--max-dist", "-m", help="Maximum radius distance for adjacency graph"),
    tile_size: float = typer.Option(50.0, "--tile-size", "-t", help="Tile size for out-of-core processing"),
    n_jobs: int = typer.Option(1, "--n-jobs", "-j", help="Number of parallel worker processes (-1 for all cores)"),
    use_arrow: bool = typer.Option(False, "--use-arrow", "-a", help="Use zero-copy PyArrow engine for loading data"),
    generate_mock: bool = typer.Option(False, "--generate-mock", help="Generate mock transcripts CSV before running")
):
    """
    Run spatial transcriptomics processing pipeline.
    """
    input_file = input
    if not input_file or generate_mock:
        input_file = "mock_transcripts.csv"
        console.print("[bold yellow][!] No input specified or mock flag set. Generating mock data...[/bold yellow]")
        generate_vectorized_mock_data(input_file)
        
    console.print(f"[bold blue][*] Loading dataset: {input_file} (use_arrow={use_arrow})...[/bold blue]")
    df = load_transcript_data(input_file, use_arrow=use_arrow)
    console.print(f"[bold green][+] Loaded {len(df)} total transcript molecules.[/bold green]")
    
    console.print(f"[bold blue][*] Binning spatial grid (Resolution: {resolution})...[/bold blue]")
    sparse_counts, bin_coords = compute_square_bins(df, resolution)
    console.print(f"[bold green][+] Created {sparse_counts.shape[0]} unique active bins across {sparse_counts.shape[1]} genes.[/bold green]")
    
    console.print(f"[bold blue][*] Building spatial adjacency graph (Max Dist: {max_dist}, n_jobs: {n_jobs})...[/bold blue]")
    adj, laplacian = build_spatial_graph(bin_coords, max_dist, n_jobs=n_jobs)
    
    console.print("\n[bold green]=== Execution Summary ===[/bold green]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    table.add_row("Total Transcripts", str(len(df)))
    table.add_row("Unique Active Bins", str(sparse_counts.shape[0]))
    table.add_row("Unique Genes", str(sparse_counts.shape[1]))
    table.add_row("Graph Non-Zero Edges", str(adj.nnz))
    table.add_row("Laplacian Matrix Shape", str(laplacian.shape))
    console.print(table)

@app.command()
def benchmark(
    num_molecules: int = typer.Option(20000, "--num-molecules", "-n", help="Number of synthetic molecules for benchmarking"),
    resolution: float = typer.Option(10.0, "--resolution", "-r", help="Spatial grid resolution"),
    tile_size: float = typer.Option(50.0, "--tile-size", "-t", help="Tile size for out-of-core benchmark"),
    max_dist: float = typer.Option(15.0, "--max-dist", "-m", help="Max graph radius distance")
):
    """
    Benchmark spatial binning algorithms (Spatial Hashing vs KD-Tree vs Quadtree) and profile out-of-core peak memory usage.
    """
    mock_file = "tmp_benchmark_data.csv"
    console.print(f"[bold blue][*] Generating {num_molecules} synthetic transcripts for benchmarking...[/bold blue]")
    generate_vectorized_mock_data(mock_file, num_molecules=num_molecules)
    
    df = load_transcript_data(mock_file)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    
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

def main():
    app()

if __name__ == "__main__":
    main()
