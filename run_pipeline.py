import argparse
import os

from spatial_pipeline.binning import compute_square_bins
from spatial_pipeline.graphs import build_spatial_graph
from spatial_pipeline.io import generate_vectorized_mock_data, load_transcript_data
from spatial_pipeline.out_of_core import (
    enforce_memory_limit,
    partition_dataset_to_tiles,
    process_tiles_parallel_out_of_core,
    stitch_out_of_core_results,
)


def main():
    parser = argparse.ArgumentParser(description="Run Memory-Constrained Spatial Transcriptomics Pipeline")
    parser.add_argument("--input", type=str, help="Path to raw molecule CSV data")
    parser.add_argument("--resolution", type=float, default=10.0, help="Grid size for spatial binning")
    parser.add_argument("--max_dist", type=float, default=15.0, help="Max radius for graph neighbors")
    parser.add_argument("--tile-size", type=float, default=100.0, help="Tile size for out-of-core processing")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for graph & tile computation")
    parser.add_argument("--out-of-core", action="store_true", help="Force out-of-core disk-backed pipeline")
    parser.add_argument("--max-ram-mb", type=float, default=512.0, help="Hard RAM cap in MB")
    parser.add_argument("--generate-mock", action="store_true", help="Generate mock dataset before running")
    
    args = parser.parse_args()
    
    input_file = args.input
    
    # Fallback to mock data if input file missing or generate-mock set
    if not input_file:
        input_file = "mock_transcripts.csv"
        print("[!] No input file specified. Automatically generating mock transcript data...")
        generate_vectorized_mock_data(input_file)
    elif args.generate_mock:
        input_file = "mock_transcripts.csv"
        generate_vectorized_mock_data(input_file)
        
    file_size_bytes = os.path.getsize(input_file) if os.path.exists(input_file) else 0
    file_size_mb = file_size_bytes / (1024.0 * 1024.0)
    
    # Auto-toggle out-of-core mode if dataset > 100 MB or explicitly requested
    auto_ooc = file_size_mb > 100.0
    use_out_of_core = args.out_of_core or auto_ooc
    
    if auto_ooc:
        print(f"[!] File size ({file_size_mb:.2f} MB) exceeds 100 MB auto-threshold. Automatically toggling Out-of-Core Processing Mode.")
        
    if use_out_of_core:
        print(f"[*] Executing OUT-OF-CORE engine (RAM budget: {args.max_ram_mb} MB, tile size: {args.tile_size} um, n_jobs: {args.n_jobs})...")
        partition_dir = "tmp_partition_run"
        tmp_out_dir = "tmp_out_run"
        
        with enforce_memory_limit(args.max_ram_mb):
            print(f"[*] 1. Partitioning dataset {input_file} into spatial tiles...")
            partition_dataset_to_tiles(input_file, args.tile_size, partition_dir)
            
            tiles = []
            if os.path.exists(partition_dir):
                for f in os.listdir(partition_dir):
                    if f.startswith("tile_") and f.endswith(".csv"):
                        parts = f[5:-4].split("_")
                        tiles.append((int(parts[0]), int(parts[1])))
                        
            print(f"[*] 2. Concurrently processing {len(tiles)} tiles across worker pool...")
            process_tiles_parallel_out_of_core(
                tiles, args.tile_size, args.resolution, args.max_dist, partition_dir, tmp_out_dir, n_jobs=args.n_jobs
            )
            
            print(f"[*] 3. Stitching out-of-core spatial graph and count matrices...")
            global_expression, global_centers, global_labels, adj, laplacian = stitch_out_of_core_results(
                tmp_out_dir, args.resolution
            )
            
        print(f"[+] Out-of-Core Processing complete!")
        print(f"    - Global Binned Nodes: {global_expression.shape[0]}")
        print(f"    - Total Unique Labels:  {len(global_labels)}")
        print(f"    - Spatial Adjacency Edges: {adj.nnz}")
        print(f"    - Graph Laplacian Shape: {laplacian.shape}")
        print("[===] Memory-Constrained Engine finished processing successfully! [===]")
    else:
        print(f"[*] Ingesting data from: {input_file}...")
        df = load_transcript_data(input_file)
        print(f"[+] Loaded {len(df)} total transcript molecules.")
        
        print(f"[*] Executing spatial grid binning (Resolution: {args.resolution})...")
        sparse_counts, bin_coords = compute_square_bins(df, args.resolution)
        print(f"[+] Created {sparse_counts.shape[0]} unique active bins across {sparse_counts.shape[1]} labels.")
        
        print(f"[*] Generating spatial adjacency graph & Graph Laplacian (Max Dist: {args.max_dist}, n_jobs: {args.n_jobs})...")
        adj, laplacian = build_spatial_graph(bin_coords, args.max_dist, n_jobs=args.n_jobs)
        print(f"[+] Spatial graph construction complete!")
        print(f"    - Adjacency non-zero edges: {adj.nnz}")
        print(f"    - Graph Laplacian matrix shape: {laplacian.shape}")
        print("[===] Pipeline finished processing successfully! [===]")


if __name__ == "__main__":
    main()
