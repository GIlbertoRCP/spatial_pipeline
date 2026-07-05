import argparse
from spatial_pipeline.io import load_transcript_data, generate_vectorized_mock_data
from spatial_pipeline.binning import compute_square_bins
from spatial_pipeline.graphs import build_spatial_graph

def main():
    parser = argparse.ArgumentParser(description="Run Spatial Transcriptomics Preprocessing Pipeline")
    parser.add_argument("--input", type=str, help="Path to raw molecule CSV data")
    parser.add_argument("--resolution", type=float, default=10.0, help="Grid size for spatial binning")
    parser.add_argument("--max_dist", type=float, default=15.0, help="Max radius for graph neighbors")
    parser.add_argument("--generate-mock", action="store_true", help="Generate a mock_transcripts.csv and run the pipeline")
    
    args = parser.parse_args()
    
    input_file = args.input
    
    # Automatically fallback to mock generation if no input file is provided
    if not input_file:
        input_file = "mock_transcripts.csv"
        print("[!] No input file specified. Automatically generating mock transcript data...")
        generate_vectorized_mock_data(input_file)
    elif args.generate_mock:
        # If input was specified but the user explicitly requested mock data generation
        input_file = "mock_transcripts.csv"
        generate_vectorized_mock_data(input_file)
        
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
