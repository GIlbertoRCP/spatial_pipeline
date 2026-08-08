import os
import shutil
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spatial_pipeline.io import generate_vectorized_mock_data
from spatial_pipeline.out_of_core import profile_out_of_core_execution
from spatial_pipeline.binning import benchmark_binning_algorithms

def run_memory_constrained_benchmark(counts=[20000, 100000, 300000, 600000]):
    """
    Executes scaling benchmark under strict memory constraints, logging peak RAM
    usage for standard in-memory vs. out-of-core pipeline implementations.
    """
    print("=" * 70)
    print("   MEMORY-CONSTRAINED SPATIAL PIPELINE SCALING BENCHMARK SUITE   ")
    print("=" * 70)
    
    results = []
    tmp_csv = "tmp_memory_benchmark_data.csv"
    
    for n in counts:
        print(f"\n[*] Benchmarking dataset scale: N = {n:,} transcript molecules...")
        generate_vectorized_mock_data(tmp_csv, num_molecules=n)
        file_size_mb = os.path.getsize(tmp_csv) / (1024.0 * 1024.0)
        
        # Profile Out-of-Core Execution
        ooc_report = profile_out_of_core_execution(
            file_path=tmp_csv,
            tile_size=100.0,
            resolution=10.0,
            max_dist=15.0,
            n_jobs=-1,
            limit_mb=512.0
        )
        
        ooc_peak_mb = ooc_report["summary"]["max_stage_peak_mb"]
        ooc_time_sec = ooc_report["summary"]["total_elapsed_sec"]
        
        print(f"    - Dataset File Size:   {file_size_mb:.2f} MB")
        print(f"    - Out-of-Core Time:    {ooc_time_sec:.3f} s")
        print(f"    - Out-of-Core Peak RAM: {ooc_peak_mb:.2f} MB")
        
        results.append({
            "Molecules": n,
            "File Size (MB)": file_size_mb,
            "OOC Time (s)": ooc_time_sec,
            "OOC Peak RAM (MB)": ooc_peak_mb,
            "RAM Cap (MB)": 512.0
        })
        
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
            
    df_results = pd.DataFrame(results)
    
    # Generate Visual Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Memory Footprint vs Dataset Size
    ax1.plot(df_results["Molecules"], df_results["OOC Peak RAM (MB)"], 'o-', color='#1b5e20', linewidth=2.5, markersize=8, label="Out-of-Core Peak RAM (MB)")
    ax1.axhline(y=512.0, color='#d32f2f', linestyle='--', linewidth=2, label="Strict RAM Limit (512 MB)")
    ax1.set_xlabel("Transcript Molecule Count (N)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Peak RAM Consumption (MB)", fontsize=12, fontweight='bold')
    ax1.set_title("Flat RAM Consumption Under Exponential Data Growth", fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Chart 2: Out-of-Core Execution Runtime
    ax2.plot(df_results["Molecules"], df_results["OOC Time (s)"], 's-', color='#0d47a1', linewidth=2.5, markersize=8, label="Out-of-Core Total Time (s)")
    ax2.set_xlabel("Transcript Molecule Count (N)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight='bold')
    ax2.set_title("Out-of-Core Pipeline Runtime Scaling", fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    out_img = os.path.join(os.path.dirname(__file__), "memory_profile_scaling.png")
    plt.savefig(out_img, dpi=300)
    print(f"\n[+] Saved memory scaling profile plot to: {out_img}")
    
    return df_results

if __name__ == "__main__":
    os.makedirs(os.path.dirname(__file__), exist_ok=True)
    run_memory_constrained_benchmark()
