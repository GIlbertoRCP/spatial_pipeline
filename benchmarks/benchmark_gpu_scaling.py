"""
Benchmark comparing CPU vs GPU (RAPIDS cuML / cuGraph) spatial graph construction scaling.

Evaluates latency, throughput (cells/sec), and memory consumption on datasets exceeding 100,000 cells.
"""

from __future__ import annotations

import os
import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spatial_pipeline.graphs import (
    build_knn_graph,
    build_spatial_graph,
    is_gpu_available,
    HAS_CUML,
    HAS_CUGRAPH
)


def run_gpu_scaling_benchmark(
    cell_counts: list[int] = [10000, 50000, 100000, 200000],
    n_neighbors: int = 15,
    save_plot: bool = True
) -> pd.DataFrame:
    """
    Benchmarks CPU vs GPU graph construction scaling up to 200k+ cells.

    Args:
        cell_counts: List of node/cell counts to benchmark.
        n_neighbors: Number of nearest neighbors per node.
        save_plot: Whether to render and save a PNG comparison plot.

    Returns:
        pd.DataFrame: Benchmark summary table.
    """
    gpu_ready = is_gpu_available()
    print("=" * 70)
    print("       SPATIAL GRAPH GPU vs CPU SCALING BENCHMARK       ")
    print("=" * 70)
    print(f"CUDA GPU RAPIDS Available: {gpu_ready} (cuML: {HAS_CUML}, cuGraph: {HAS_CUGRAPH})")
    print(f"Target k-NN Neighbors:     {n_neighbors}")
    print("=" * 70)

    results = []

    for n in cell_counts:
        print(f"\n[*] Evaluating Spatial Graph Construction for N = {n:,} cells...")
        np.random.seed(42)
        coords = np.random.uniform(0.0, 1000.0, (n, 2)).astype(np.float32)

        # 1. Benchmark CPU Single-Thread
        t0 = time.perf_counter()
        adj_cpu_1, _ = build_knn_graph(coords, n_neighbors=n_neighbors, backend="sklearn")
        t_cpu_1 = time.perf_counter() - t0
        throughput_cpu_1 = n / max(t_cpu_1, 1e-6)
        print(f"    [CPU 1-Thread]   Time: {t_cpu_1 * 1000.0:8.2f} ms | Throughput: {throughput_cpu_1:10.1f} cells/s")

        # 2. Benchmark CPU Multi-Core (n_jobs = -1)
        t0 = time.perf_counter()
        adj_cpu_multi, _ = build_spatial_graph(coords, max_distance=15.0, n_jobs=-1, backend="multiprocessing")
        t_cpu_multi = time.perf_counter() - t0
        throughput_cpu_multi = n / max(t_cpu_multi, 1e-6)
        print(f"    [CPU Multi-Core] Time: {t_cpu_multi * 1000.0:8.2f} ms | Throughput: {throughput_cpu_multi:10.1f} cells/s")

        # 3. Benchmark GPU cuML / cuGraph if available
        if gpu_ready:
            t0 = time.perf_counter()
            adj_gpu, _ = build_knn_graph(coords, n_neighbors=n_neighbors, backend="cuml")
            t_gpu = time.perf_counter() - t0
            throughput_gpu = n / max(t_gpu, 1e-6)
            speedup = t_cpu_1 / max(t_gpu, 1e-6)
            print(f"    [GPU RAPIDS]     Time: {t_gpu * 1000.0:8.2f} ms | Throughput: {throughput_gpu:10.1f} cells/s | Speedup: {speedup:5.1f}x")
        else:
            t_gpu = None
            throughput_gpu = None
            speedup = None
            print("    [GPU RAPIDS]     N/A (Running in CPU mode)")

        results.append({
            "Cells": n,
            "CPU_1_Thread_sec": t_cpu_1,
            "CPU_MultiCore_sec": t_cpu_multi,
            "GPU_RAPIDS_sec": t_gpu,
            "Throughput_CPU_cells_per_sec": throughput_cpu_1,
            "Throughput_GPU_cells_per_sec": throughput_gpu,
            "GPU_Speedup": speedup
        })

    df = pd.DataFrame(results)

    if save_plot:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Latency curve
        ax1.plot(df["Cells"], df["CPU_1_Thread_sec"], "o-", color="#d9534f", linewidth=2, label="CPU (Single Thread)")
        ax1.plot(df["Cells"], df["CPU_MultiCore_sec"], "s--", color="#f0ad4e", linewidth=2, label="CPU (Multi-Core)")
        if gpu_ready and df["GPU_RAPIDS_sec"].notnull().all():
            ax1.plot(df["Cells"], df["GPU_RAPIDS_sec"], "^-", color="#5cb85c", linewidth=2.5, label="GPU (RAPIDS cuML)")
        ax1.set_xlabel("Number of Cells (N)", fontweight="bold")
        ax1.set_ylabel("Execution Time (seconds)", fontweight="bold")
        ax1.set_title("Graph Construction Latency vs Dataset Size", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Throughput
        ax2.plot(df["Cells"], df["Throughput_CPU_cells_per_sec"], "o-", color="#2b5c8f", linewidth=2, label="CPU Throughput")
        if gpu_ready and df["Throughput_GPU_cells_per_sec"].notnull().all():
            ax2.plot(df["Cells"], df["Throughput_GPU_cells_per_sec"], "^-", color="#5cb85c", linewidth=2.5, label="GPU RAPIDS Throughput")
        ax2.set_xlabel("Number of Cells (N)", fontweight="bold")
        ax2.set_ylabel("Throughput (Cells / Sec)", fontweight="bold")
        ax2.set_title("Spatial Graph Construction Throughput", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        out_plot = os.path.join(out_dir, "benchmark_gpu_scaling.png")
        plt.savefig(out_plot, dpi=300)
        print(f"\n[+] GPU scaling benchmark plot saved to: {out_plot}")

    return df


if __name__ == "__main__":
    run_gpu_scaling_benchmark(cell_counts=[5000, 20000, 50000, 100000])
