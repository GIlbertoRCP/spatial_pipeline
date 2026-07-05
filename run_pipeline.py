import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from spatial_pipeline.io import load_transcript_data
from spatial_pipeline.binning import compute_square_bins
from spatial_pipeline.graphs import build_spatial_graph

def generate_mock_data(file_path: str, num_molecules: int = 15000) -> None:
    """
    Generates a synthetic spatial transcriptomics CSV file for out-of-the-box pipeline runs.
    """
    print(f"[*] Generating synthetic mock dataset ({num_molecules} molecules)...")
    np.random.seed(42)
    x = np.random.uniform(10.0, 500.0, num_molecules)
    y = np.random.uniform(10.0, 500.0, num_molecules)
    
    # Simulate spatial aggregation in clusters to make the graph interesting
    n_clusters = 5
    cluster_centers = np.random.uniform(50.0, 450.0, (n_clusters, 2))
    for i in range(n_clusters):
        cx, cy = cluster_centers[i]
        c_size = num_molecules // (n_clusters * 2)
        x = np.concatenate([x, np.random.normal(cx, 15.0, c_size)])
        y = np.concatenate([y, np.random.normal(cy, 15.0, c_size)])
        
    x = np.clip(x, 10.0, 500.0)
    y = np.clip(y, 10.0, 500.0)
    
    genes = np.random.choice(["GAPDH", "ACTB", "MALAT1", "EPCAM", "CD3D"], size=len(x))
    
    df = pd.DataFrame({
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "gene": genes
    })
    
    df.to_csv(file_path, index=False)
    print(f"[+] Automated mock dataset successfully saved to: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Spatial Transcriptomics Preprocessing Pipeline")
    parser.add_argument("--input", type=str, help="Path to raw molecule CSV data")
    parser.add_argument("--resolution", type=float, default=10.0, help="Grid size for spatial binning")
    parser.add_argument("--max_dist", type=float, default=15.0, help="Max radius for graph neighbors")
    parser.add_argument("--generate-mock", action="store_true", help="Generate a mock_transcripts.csv and run the pipeline")
    
    args = parser.parse_args()
    
    if not args.input and not args.generate_mock:
        parser.error("one of the arguments --input or --generate-mock is required")
        
    input_file = args.input
    if args.generate_mock:
        input_file = "mock_transcripts.csv"
        generate_mock_data(input_file)
        
    print(f"[*] Ingesting data from: {input_file}...")
    df = load_transcript_data(input_file)
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
