"""
Quickstart Tutorial: Connecting spatial_pipeline to Scanpy & Squidpy
====================================================================
This tutorial demonstrates end-to-end spatial processing:
1. Generating raw spatial transcriptomics coordinates.
2. Running vectorized spatial binning and multi-core graph Laplacian construction.
3. Exporting to standard AnnData (.h5ad) format.
4. Performing spatial bioinformatics analysis.
"""

import os

import numpy as np

from spatial_pipeline import (
    build_spatial_graph,
    compute_square_bins,
    generate_vectorized_mock_data,
    load_transcript_data,
    save_anndata,
    to_anndata,
)


def main():
    print("=" * 65)
    print("   TUTORIAL: SPATIAL PIPELINE TO SCANPY/SQUIDPY ANNDATA INTEROP   ")
    print("=" * 65)
    
    # 1. Generate synthetic transcripts CSV
    csv_path = "sample_transcripts.csv"
    print("\n[*] Step 1: Generating 25,000 synthetic transcripts...")
    generate_vectorized_mock_data(csv_path, num_molecules=25000)
    
    # 2. Ingest transcripts using low-memory datatypes
    print("\n[*] Step 2: Loading transcript coordinates...")
    df = load_transcript_data(csv_path)
    print(f"[+] Loaded {len(df):,} total molecules across {len(df['gene'].cat.categories)} unique genes.")
    
    # 3. Vectorized spatial binning (15 um resolution grid, min count threshold = 2)
    print("\n[*] Step 3: Executing spatial grid binning (Resolution: 15.0 um)...")
    sparse_counts, bin_coords = compute_square_bins(df, resolution=15.0, min_counts=2)
    print(f"[+] Created {sparse_counts.shape[0]:,} active spatial bins (Matrix shape: {sparse_counts.shape}).")
    
    # 4. Construct spatial adjacency graph and Graph Laplacian (Max distance: 25.0 um, all CPU cores)
    print("\n[*] Step 4: Building spatial adjacency graph (Max Radius: 25.0 um, n_jobs=-1)...")
    adj_matrix, laplacian_matrix = build_spatial_graph(bin_coords, max_distance=25.0, n_jobs=-1)
    print(f"[+] Constructed topological spatial graph with {adj_matrix.nnz // 2:,} undirected edges.")
    
    # 5. Export to AnnData (.h5ad) following Scanpy/Squidpy standards
    print("\n[*] Step 5: Exporting to Scanpy/Squidpy AnnData object...")
    adata = to_anndata(
        sparse_counts,
        bin_coords,
        genes=np.array(df["gene"].cat.categories),
        adjacency_matrix=adj_matrix,
        laplacian_matrix=laplacian_matrix,
        metadata={"sample_id": "sample_001", "resolution_um": 15.0}
    )
    
    print("\n[+] AnnData Object Structure:")
    print(adata)
    print(f"    - Expression Matrix (adata.X): {adata.X.shape}")
    print(f"    - Spatial Coordinates (adata.obsm['spatial']): {adata.obsm['spatial'].shape}")
    print(f"    - Adjacency Graph (adata.obsp['spatial_connectivities']): {adata.obsp['spatial_connectivities'].shape}")
    print(f"    - Graph Laplacian (adata.obsp['spatial_laplacian']): {adata.obsp['spatial_laplacian'].shape}")
    
    # 6. Save as standard HDF5 .h5ad file
    h5ad_out = "sample_spatial_processed.h5ad"
    save_anndata(adata, h5ad_out)
    print(f"\n[+] Successfully saved AnnData file: {h5ad_out} ({os.path.getsize(h5ad_out) / 1024:.1f} KB)")
    
    # Cleanup temporary CSV
    if os.path.exists(csv_path):
        os.remove(csv_path)

if __name__ == "__main__":
    main()
