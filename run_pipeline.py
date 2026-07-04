import argparse
import numpy as np
from spatial_pipeline.io import load_transcript_data
from spatial_pipeline.binning import compute_square_bins
from spatial_pipeline.graphs import build_spatial_graph

def main():
    parser = argparse.ArgumentParser(description="Run Spatial Transcriptomics Preprocessing Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to raw molecule CSV data")
    parser.add_argument("--resolution", type=float, default=10.0, help="Grid size for spatial binning")
    parser.add_argument("--max_dist", type=float, default=15.0, help="Max radius for graph neighbors")
    
    args = parser.parse_args()
    
    print(f"[*] Ingesting data from: {args.input}...")
    df = load_transcript_data(args.input)
    print(f"[+] Loaded {len(df)} total transcript molecules.")
    
    print(f"[*] Executing spatial grid binning (Resolution: {args.resolution})...")
    sparse_counts, bin_coords = compute_square_bins(df, args.resolution)
    print(f"[+] Created {sparse_counts.shape[0]} unique active bins across {sparse_counts.shape[1]} genes.")
    
    print(f"[*] Generating spatial adjacency graph & Graph Laplacian (Max Dist: {args.max_dist})...")
    adj, laplacian = build_spatial_graph(bin_coords, args.max_dist)
    print(f"[+] Spatial graph construction complete!")
    print(f"    - Adjacency non-zero edges: {adj.nnz}")
    print(f"    - Graph Laplacian matrix shape: {laplacian.shape}")
    print("[===] Pipeline finished processing successfully! [===]")

if __name__ == "__main__":
    main()
