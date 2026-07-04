import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from src.spatial_pipeline.io import download_dataset, load_coordinates
from src.spatial_pipeline.binning import (
    bin_square, 
    bin_hexagonal, 
    get_bin_centers, 
    build_sparse_matrix
)
from src.spatial_pipeline.graphs import build_adjacency_graph

def generate_mock_dataset(file_path: str) -> None:
    """
    Generates a realistic synthetic spatial transcriptomics dataset for testing.
    """
    print("Generating synthetic spatial transcriptomics dataset...")
    np.random.seed(42)
    n_molecules = 150000
    
    # Base grid
    x = np.random.uniform(100.0, 1100.0, n_molecules)
    y = np.random.uniform(100.0, 1100.0, n_molecules)
    
    # Add biological clusters (simulate cell-like densities or tissue regions)
    n_clusters = 40
    cluster_centers = np.random.uniform(150.0, 1050.0, (n_clusters, 2))
    cluster_weights = np.random.exponential(100.0, n_clusters)
    cluster_weights /= cluster_weights.sum()
    
    assigned_clusters = np.random.choice(n_clusters, size=n_molecules, p=cluster_weights)
    x_noise = np.random.normal(0.0, 20.0, n_molecules)
    y_noise = np.random.normal(0.0, 20.0, n_molecules)
    x = cluster_centers[assigned_clusters, 0] + x_noise
    y = cluster_centers[assigned_clusters, 1] + y_noise
    
    x = np.clip(x, 100.0, 1100.0)
    y = np.clip(y, 100.0, 1100.0)
    
    # Gene panel
    genes_list = [
        "EPCAM", "CD3D", "MS4A1", "CD8A", "CD4", "ERBB2", "ESR1", "PGR", "GAPDH", "ACTB",
        "KRT5", "KRT8", "KRT18", "KRT19", "PECAM1", "COL1A1", "FN1", "VIM", "B2M", "EEF1A1"
    ]
    gene_probs = np.random.dirichlet(np.ones(len(genes_list)))
    genes = np.random.choice(genes_list, size=n_molecules, p=gene_probs)
    
    # Q-values
    qv = np.random.exponential(8.0, n_molecules) + np.random.uniform(5.0, 25.0, n_molecules)
    
    df = pd.DataFrame({
        "x_location": x,
        "y_location": y,
        "feature_name": genes,
        "qv": qv
    })
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False, compression="gzip")
    print(f"Synthetic dataset saved to {file_path}")

def plot_binned_expression(
    bin_centers: np.ndarray, 
    counts: np.ndarray, 
    bin_type: str, 
    resolution: float, 
    output_path: str
) -> None:
    """
    Plots the spatial distribution of binned transcript counts.
    """
    print(f"Generating spatial plot and saving to {output_path}...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if bin_type == "square":
        # Render square bins as scatter points or colored rectangles
        # Scatter with square marker is quick and effective
        scatter = ax.scatter(
            bin_centers[:, 0], 
            bin_centers[:, 1], 
            c=counts, 
            cmap="viridis", 
            s=(resolution * 0.8) ** 2,  # scale marker size roughly with resolution
            marker="s",
            edgecolors="none"
        )
        plt.colorbar(scatter, ax=ax, label="Total Transcripts")
        
    elif bin_type == "hex":
        # Draw hexagons using PolyCollection for professional visual styling
        R = resolution / np.sqrt(3.0)
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        hex_offset = np.column_stack((R * np.cos(angles), R * np.sin(angles)))
        
        # Build hex vertices for all centers
        polygons = [bin_centers[i] + hex_offset for i in range(len(bin_centers))]
        
        coll = PolyCollection(
            polygons, 
            array=counts, 
            cmap="viridis", 
            edgecolors="face"
        )
        ax.add_collection(coll)
        ax.autoscale_view()
        plt.colorbar(coll, ax=ax, label="Total Transcripts")
        
    ax.set_aspect("equal")
    ax.set_title(f"Binned Spatial Transcriptomics ({bin_type.capitalize()} Grid, spacing={resolution})")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Fast Sparse Matrix and Spatial Binning Pipeline for Spatial Transcriptomics."
    )
    parser.add_argument(
        "--input", 
        type=str, 
        help="Path to the transcripts input CSV/compressed CSV/Parquet file."
    )
    parser.add_argument(
        "--bin-type", 
        choices=["square", "hex"], 
        default="square", 
        help="Type of grid pattern for spatial binning: 'square' or 'hex' (default: square)."
    )
    parser.add_argument(
        "--resolution", 
        type=float, 
        default=20.0, 
        help="Spatial resolution representing spacing between adjacent bin centers (default: 20.0)."
    )
    parser.add_argument(
        "--min-qv", 
        type=float, 
        default=20.0, 
        help="Minimum quality value (QV) threshold for loaded transcripts (default: 20.0)."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output", 
        help="Directory to save generated outputs (default: output)."
    )
    parser.add_argument(
        "--generate-mock", 
        action="store_true", 
        help="Generate and run the pipeline on a synthetic mock dataset."
    )
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Download and use the public 10x Genomics Xenium Human Breast 2fov sample dataset."
    )
    
    args = parser.parse_args()
    
    # 1. Establish the input path based on user choice
    input_path = args.input
    
    if args.generate_mock:
        input_path = "data/mock_transcripts.csv.gz"
        generate_mock_dataset(input_path)
    elif args.download:
        url = "https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Breast_2fov/Xenium_V1_human_Breast_2fov_outs.zip"
        download_dir = "data"
        extracted_dir = download_dataset(url, download_dir)
        
        # Check inside extracted directories for transcripts file
        possible_paths = [
            os.path.join(extracted_dir, "transcripts.csv.gz"),
            os.path.join(extracted_dir, "transcripts.parquet"),
            os.path.join(extracted_dir, "Xenium_V1_human_Breast_2fov_outs", "transcripts.csv.gz"),
            os.path.join(extracted_dir, "Xenium_V1_human_Breast_2fov_outs", "transcripts.parquet"),
        ]
        
        found = False
        for path in possible_paths:
            if os.path.exists(path):
                input_path = path
                found = True
                break
                
        if not found:
            raise FileNotFoundError(
                f"Could not locate transcripts.csv.gz or transcripts.parquet inside extracted folder {extracted_dir}"
            )
            
    if not input_path:
        parser.print_help()
        print("\nError: Please provide --input, run with --generate-mock, or run with --download.")
        return
        
    # 2. Load dataset
    df = load_coordinates(input_path, min_qv=args.min_qv)
    x = df["x"].values
    y = df["y"].values
    genes = df["gene"].values
    
    # 3. Compute binning grid
    print(f"Binning spatial transcripts using {args.bin_type} grid with resolution {args.resolution}...")
    if args.bin_type == "square":
        bin_coords, offset = bin_square(x, y, args.resolution)
    else:  # hex
        bin_coords = bin_hexagonal(x, y, args.resolution)
        offset = None
        
    # 4. Generate sparse expression count matrix
    print("Building sparse expression count matrix...")
    matrix, unique_bins, unique_genes = build_sparse_matrix(bin_coords, genes, "csr")
    
    # 5. Compute bin center Cartesian coordinates
    bin_centers = get_bin_centers(unique_bins, args.bin_type, args.resolution, offset)
    
    # 6. Construct spatial adjacency graph
    print("Generating spatial adjacency graph...")
    adj_matrix = build_adjacency_graph(bin_centers, args.bin_type, args.resolution)
    
    # 7. Print pipeline analytics
    n_transcripts = len(df)
    n_bins = len(unique_bins)
    n_genes = len(unique_genes)
    sparsity = 100.0 * (1.0 - matrix.nnz / (n_bins * n_genes)) if (n_bins * n_genes) > 0 else 0.0
    n_edges = adj_matrix.nnz // 2  # Undirected edges
    
    print("\n--- Pipeline Run Summary ---")
    print(f"Loaded transcripts: {n_transcripts:,}")
    print(f"Unique active bins: {n_bins:,}")
    print(f"Unique genes:       {n_genes:,}")
    print(f"Matrix shape:       {matrix.shape}")
    print(f"Matrix sparsity:    {sparsity:.2f}%")
    print(f"Adjacency edges:    {n_edges:,}")
    print("----------------------------\n")
    
    # 8. Save output files
    os.makedirs(args.output_dir, exist_ok=True)
    
    expr_path = os.path.join(args.output_dir, "expression_matrix.npz")
    scipy.sparse.save_npz(expr_path, matrix)
    
    graph_path = os.path.join(args.output_dir, "adjacency_matrix.npz")
    scipy.sparse.save_npz(graph_path, adj_matrix)
    
    centers_path = os.path.join(args.output_dir, "bin_centers.npy")
    np.save(centers_path, bin_centers)
    
    genes_path = os.path.join(args.output_dir, "genes.npy")
    np.save(genes_path, unique_genes)
    
    coords_path = os.path.join(args.output_dir, "bin_coords.npy")
    np.save(coords_path, unique_bins)
    
    if offset is not None:
        np.save(os.path.join(args.output_dir, "offset.npy"), np.array(offset))
        
    print(f"Pipeline outputs successfully saved to directory: '{args.output_dir}'")
    
    # 9. Plot visualization
    total_counts_per_bin = np.array(matrix.sum(axis=1)).flatten()
    plot_path = os.path.join(args.output_dir, "spatial_binned_plot.png")
    plot_binned_expression(bin_centers, total_counts_per_bin, args.bin_type, args.resolution, plot_path)
    print("Pipeline process completed successfully.")

if __name__ == "__main__":
    main()
