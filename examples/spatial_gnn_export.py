"""
Tutorial: Exporting Spatial Binning Graphs to PyTorch Geometric (PyG) for Spatial GNNs
====================================================================================
This script demonstrates:
1. Spatial transcriptomics coordinate binning and graph construction.
2. Direct export to PyTorch Geometric data format (`to_pyg_data()`).
3. Computing Graph Laplacian eigenvalue spectrum for spatial domain analysis.
"""

import os

import numpy as np

from spatial_pipeline import (
    build_spatial_graph,
    compute_laplacian_spectrum,
    compute_square_bins,
    generate_vectorized_mock_data,
    load_transcript_data,
    to_pyg_data,
)


def main():
    print("=" * 65)
    print("   TUTORIAL: SPATIAL GNN DATA EXPORT & LAPLACIAN SPECTRUM   ")
    print("=" * 65)
    
    # 1. Generate synthetic spatial data
    csv_file = "gnn_transcripts_sample.csv"
    print("\n[*] Step 1: Generating synthetic transcript coordinates...")
    generate_vectorized_mock_data(csv_file, num_molecules=15000)
    
    # 2. Bin coordinates and build spatial graph
    df = load_transcript_data(csv_file)
    print("\n[*] Step 2: Binning spatial coordinates (Resolution = 20.0 um)...")
    sparse_counts, bin_coords = compute_square_bins(df, resolution=20.0)
    
    print("\n[*] Step 3: Constructing spatial graph (Max distance = 35.0 um)...")
    adj_matrix, laplacian_matrix = build_spatial_graph(bin_coords, max_distance=35.0, n_jobs=-1)
    
    # 3. Export to PyTorch Geometric format
    print("\n[*] Step 4: Converting spatial graph to PyTorch Geometric (PyG) format...")
    pyg_obj = to_pyg_data(sparse_counts, bin_coords, adj_matrix)
    
    if isinstance(pyg_obj, dict):
        print("[+] PyG-Compatible Standard Dictionary:")
        print(f"    - Node Features (x): {pyg_obj['x'].shape} (num_bins x num_genes)")
        print(f"    - Spatial Coordinates (pos): {pyg_obj['pos'].shape} (num_bins x 2)")
        print(f"    - Edge Index (edge_index): {pyg_obj['edge_index'].shape} (2 x num_edges)")
    else:
        print("[+] PyTorch Geometric Data Object:")
        print(pyg_obj)
        print(f"    - x: {pyg_obj.x.shape}")
        print(f"    - pos: {pyg_obj.pos.shape}")
        print(f"    - edge_index: {pyg_obj.edge_index.shape}")
        
    # 4. Compute Graph Laplacian Spectrum
    print("\n[*] Step 5: Computing Top 10 Smallest Graph Laplacian Eigenvalues...")
    evals = compute_laplacian_spectrum(laplacian_matrix, k=10)
    print("    Smallest Eigenvalues (lambda):", np.round(evals, 4))
    if len(evals) > 1:
        print(f"    - Graph Algebraic Connectivity (Fiedler Value lambda2): {evals[1]:.4f}")

        
    # Cleanup
    if os.path.exists(csv_file):
        os.remove(csv_file)

if __name__ == "__main__":
    main()
