import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spatial_pipeline.io import generate_vectorized_mock_data, load_transcript_data
from spatial_pipeline.binning import compute_square_bins
from spatial_pipeline.graphs import build_spatial_graph

def run_scaling_benchmark(counts=[10000, 50000, 200000, 500000]):
    print("=" * 60)
    print("      SPATIAL PIPELINE SCALING BENCHMARK SUITE      ")
    print("=" * 60)
    
    results = []
    tmp_csv = "tmp_scaling_benchmark.csv"
    
    for n in counts:
        print(f"\n[*] Benchmarking pipeline for N = {n:,} molecules...")
        generate_vectorized_mock_data(tmp_csv, num_molecules=n)
        
        # 1. Measure Ingestion
        tracemalloc.start()
        t0 = time.perf_counter()
        df = load_transcript_data(tmp_csv)
        t_ingest = (time.perf_counter() - t0) * 1000.0
        _, peak_ingest = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 2. Measure Binning
        tracemalloc.start()
        t0 = time.perf_counter()
        sparse_counts, bin_coords = compute_square_bins(df, resolution=10.0)
        t_bin = (time.perf_counter() - t0) * 1000.0
        _, peak_bin = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 3. Measure Spatial Graph Construction
        tracemalloc.start()
        t0 = time.perf_counter()
        adj, lap = build_spatial_graph(bin_coords, max_distance=15.0, n_jobs=-1)
        t_graph = (time.perf_counter() - t0) * 1000.0
        _, peak_graph = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        total_time_sec = (t_ingest + t_bin + t_graph) / 1000.0
        max_peak_mb = max(peak_ingest, peak_bin, peak_graph) / (1024.0 * 1024.0)
        
        print(f"    - Ingestion Time:  {t_ingest:.2f} ms")
        print(f"    - Binning Time:    {t_bin:.2f} ms")
        print(f"    - Graph Build Time:{t_graph:.2f} ms")
        print(f"    - Total Time:      {total_time_sec:.3f} s")
        print(f"    - Peak RAM Usage:  {max_peak_mb:.2f} MB")
        
        results.append({
            "Molecules": n,
            "Ingestion (ms)": t_ingest,
            "Binning (ms)": t_bin,
            "Graph (ms)": t_graph,
            "Total Time (s)": total_time_sec,
            "Peak RAM (MB)": max_peak_mb,
            "Active Bins": sparse_counts.shape[0]
        })
        
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
            
    df_results = pd.DataFrame(results)
    
    # Generate Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Runtime Scaling
    ax1.plot(df_results["Molecules"], df_results["Total Time (s)"], 'o-', color='#2b5c8f', linewidth=2.5, markersize=8, label="Total Time (s)")
    ax1.plot(df_results["Molecules"], df_results["Graph (ms)"] / 1000.0, 's--', color='#e65c00', linewidth=1.8, label="Graph Construction (s)")
    ax1.set_xlabel("Molecules (N)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight='bold')
    ax1.set_title("Pipeline Runtime Scaling vs Coordinate Count", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Plot 2: Memory Footprint
    ax2.plot(df_results["Molecules"], df_results["Peak RAM (MB)"], '^-', color='#2e7d32', linewidth=2.5, markersize=8, label="Peak Memory (MB)")
    ax2.set_xlabel("Molecules (N)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Peak RAM Usage (MB)", fontsize=12, fontweight='bold')
    ax2.set_title("Peak RAM Scaling vs Coordinate Count", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    out_img = os.path.join(os.path.dirname(__file__), "benchmark_scaling.png")
    plt.savefig(out_img, dpi=300)
    print(f"\n[+] Benchmark scaling chart saved to: {out_img}")
    return df_results

if __name__ == "__main__":
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    run_scaling_benchmark()
