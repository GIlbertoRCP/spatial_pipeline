import os
import shutil
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix, coo_matrix
from scipy.spatial import KDTree
from typing import Tuple, List, Dict, Any

def partition_dataset_to_tiles(file_path: str, tile_size: float, partition_dir: str) -> None:
    """
    Partitions a large spatial transcriptomics CSV file into spatial tiles on disk.
    
    Parameters
    ----------
    file_path : str
        Path to the raw transcript coordinates CSV.
    tile_size : float
        The physical dimensions (width/height) of each spatial tile.
    partition_dir : str
        Directory where tile partition CSV files will be stored.
    """
    if os.path.exists(partition_dir):
        # Clean existing CSV partitions to prevent appending to stale data
        for f in os.listdir(partition_dir):
            if f.endswith(".csv") or f.endswith(".npy"):
                try:
                    os.remove(os.path.join(partition_dir, f))
                except OSError:
                    pass
    else:
        os.makedirs(partition_dir, exist_ok=True)
        
    print(f"[*] Partitioning dataset {file_path} into {tile_size} um spatial tiles...")
    
    global_genes_set = set()
    x_min = float('inf')
    y_min = float('inf')
    
    # Process in row-chunks to minimize memory consumption
    for chunk in pd.read_csv(file_path, chunksize=100000):
        # Update offsets globally
        x_min = min(x_min, chunk['x'].min())
        y_min = min(y_min, chunk['y'].min())
        
        # Collect all unique genes in the dataset vectorially
        global_genes_set.update(chunk['gene'].unique())
        
        # Ensure coordinates are float and calculate tile grid coordinates
        tx = np.floor(chunk['x'].to_numpy() / tile_size).astype(int)
        ty = np.floor(chunk['y'].to_numpy() / tile_size).astype(int)
        
        # Group and append each sub-chunk to its corresponding tile file
        for (tile_x, tile_y), group in chunk.groupby([tx, ty]):
            tile_file = os.path.join(partition_dir, f"tile_{tile_x}_{tile_y}.csv")
            # Write header only if file is newly created
            group.to_csv(
                tile_file, 
                mode='a', 
                index=False, 
                header=not os.path.exists(tile_file)
            )
            
    # Save global offset and gene list to the partition directory
    global_genes = np.array(sorted(list(global_genes_set)))
    np.save(os.path.join(partition_dir, "genes.npy"), global_genes)
    np.save(os.path.join(partition_dir, "offset.npy"), np.array([x_min, y_min]))
    
    print(f"[+] Dataset partitioning complete. Saved in directory: {partition_dir}")

def load_tile_with_halo(
    tile_x: int, 
    tile_y: int, 
    tile_size: float, 
    halo_width: float, 
    partition_dir: str
) -> pd.DataFrame:
    """
    Loads coordinate data for a target tile and its 8 neighboring tiles, 
    filtering to the expanded bounding box (core + halo).
    
    Parameters
    ----------
    tile_x : int
        Grid column coordinate of the target tile.
    tile_y : int
        Grid row coordinate of the target tile.
    tile_size : float
        Physical width/height of a tile.
    halo_width : float
        Width of the boundary exchange halo (typically max_dist).
    partition_dir : str
        Directory containing the tile partition CSV files.
        
    Returns
    -------
    pd.DataFrame
        DataFrame of transcripts inside the core tile + halo boundary, including source tile info.
    """
    dfs = []
    # Search the 3x3 grid of tiles centered at (tile_x, tile_y)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx = tile_x + dx
            ny = tile_y + dy
            tile_file = os.path.join(partition_dir, f"tile_{nx}_{ny}.csv")
            if os.path.exists(tile_file):
                sub_df = pd.read_csv(tile_file)
                sub_df['src_tile_x'] = nx
                sub_df['src_tile_y'] = ny
                dfs.append(sub_df)
                
    if not dfs:
        return pd.DataFrame(columns=['x', 'y', 'gene', 'src_tile_x', 'src_tile_y'])
        
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Establish expanded spatial bounds (core tile box + halo border)
    x_min = tile_x * tile_size - halo_width
    x_max = (tile_x + 1) * tile_size + halo_width
    y_min = tile_y * tile_size - halo_width
    y_max = (tile_y + 1) * tile_size + halo_width
    
    # Filter molecules to bounds
    in_bounds = (
        (df_combined['x'] >= x_min) & (df_combined['x'] <= x_max) &
        (df_combined['y'] >= y_min) & (df_combined['y'] <= y_max)
    )
    
    df_filtered = df_combined[in_bounds].copy()
    return df_filtered

def process_tile_out_of_core(
    tile_x: int, 
    tile_y: int, 
    tile_size: float, 
    resolution: float, 
    max_dist: float, 
    partition_dir: str, 
    tmp_out_dir: str
) -> None:
    """
    Bins and generates sub-graphs for a single spatial tile out-of-core.
    Active bins are classified into 'owned' or 'halo' to avoid duplications.
    """
    # Load global unique genes and offset
    genes_path = os.path.join(partition_dir, "genes.npy")
    offset_path = os.path.join(partition_dir, "offset.npy")
    if not os.path.exists(genes_path) or not os.path.exists(offset_path):
        raise FileNotFoundError("Global partition metadata not found. Run partitioning first.")
    unique_genes = np.load(genes_path, allow_pickle=True)
    x_min, y_min = np.load(offset_path)
    
    # 1. Ingest core coordinates + halo
    df = load_tile_with_halo(tile_x, tile_y, tile_size, max_dist, partition_dir)
    if len(df) == 0:
        return
        
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    
    # 2. Bin molecules relative to the global offset
    bin_x = np.floor((x - x_min) / resolution).astype(np.int32)
    bin_y = np.floor((y - y_min) / resolution).astype(np.int32)
    bin_coords = np.column_stack((bin_x, bin_y))
    
    # Unique active bins in this tile + halo
    unique_bins, inverse_indices = np.unique(bin_coords, axis=0, return_inverse=True)
    
    # Compute centers for all unique active bins using the global offset
    bin_centers = np.column_stack((
        x_min + (unique_bins[:, 0] + 0.5) * resolution,
        y_min + (unique_bins[:, 1] + 0.5) * resolution
    ))
    
    # 3. Classify which bins are owned by the current tile.
    # We assign ownership deterministically:
    # - If the bin center sits inside an active tile core, that tile is the owner.
    # - If the bin center sits inside an empty tile (no files), the active tile containing the
    #   lexicographically smallest (x, y) coordinates with transcripts for this bin is the owner.
    cx = bin_centers[:, 0]
    cy = bin_centers[:, 1]
    
    home_tile_x = np.floor(cx / tile_size).astype(np.int32)
    home_tile_y = np.floor(cy / tile_size).astype(np.int32)
    
    home_active = np.array([
        os.path.exists(os.path.join(partition_dir, f"tile_{htx}_{hty}.csv"))
        for htx, hty in zip(home_tile_x, home_tile_y)
    ])
    
    src_tile_x = df['src_tile_x'].to_numpy()
    src_tile_y = df['src_tile_y'].to_numpy()
    src_keys = src_tile_x.astype(np.int64) * 1000000 + src_tile_y.astype(np.int64)
    
    min_src_keys = np.full(len(unique_bins), 999999999, dtype=np.int64)
    np.minimum.at(min_src_keys, inverse_indices, src_keys)
    
    u_key = int(tile_x) * 1000000 + int(tile_y)
    
    is_owned = (
        (home_active & (home_tile_x == tile_x) & (home_tile_y == tile_y)) |
        (~home_active & (min_src_keys == u_key))
    )
    
    num_owned = np.sum(is_owned)
    if num_owned == 0:
        return
        
    # Get subset of owned bins and transcripts
    owned_bins = unique_bins[is_owned]
    owned_transcripts_mask = is_owned[inverse_indices]
    df_owned = df[owned_transcripts_mask]
    
    # Remap owned transcripts to the owned_bins index
    _, inverse_owned = np.unique(bin_coords[owned_transcripts_mask], axis=0, return_inverse=True)
    
    # Build count matrix for owned bins using globally aligned categorical codes
    gene_series = pd.Categorical(df_owned['gene'], categories=unique_genes)
    gene_codes = gene_series.codes
    
    counts = np.ones(len(df_owned), dtype=np.int32)
    sparse_counts = coo_matrix(
        (counts, (inverse_owned, gene_codes)),
        shape=(len(owned_bins), len(unique_genes)),
        dtype=np.int32
    ).tocsr()
    
    # 4. Generate local adjacency edges using KDTree
    tree = KDTree(bin_centers)
    neighbor_pairs = tree.query_pairs(r=max_dist, output_type='ndarray')
    
    if len(neighbor_pairs) > 0:
        # Connect bins A and B. Keep edge if A or B is owned by this tile
        i_owned = is_owned[neighbor_pairs[:, 0]]
        j_owned = is_owned[neighbor_pairs[:, 1]]
        valid_edges_mask = i_owned | j_owned
        tile_edges = neighbor_pairs[valid_edges_mask]
        
        # Save edges using their global grid coordinate identifiers (gx_A, gy_A, gx_B, gy_B)
        global_edges = np.column_stack((
            unique_bins[tile_edges[:, 0]],
            unique_bins[tile_edges[:, 1]]
        ))
    else:
        global_edges = np.empty((0, 4), dtype=np.int32)
        
    # Flush intermediate results to temporary directory
    os.makedirs(tmp_out_dir, exist_ok=True)
    prefix = f"{tile_x}_{tile_y}"
    
    scipy.sparse.save_npz(os.path.join(tmp_out_dir, f"counts_{prefix}.npz"), sparse_counts)
    np.save(os.path.join(tmp_out_dir, f"bins_{prefix}.npy"), owned_bins)
    np.save(os.path.join(tmp_out_dir, f"genes_{prefix}.npy"), unique_genes)
    np.save(os.path.join(tmp_out_dir, f"edges_{prefix}.npy"), global_edges)
    np.save(os.path.join(tmp_out_dir, "offset.npy"), np.array([x_min, y_min]))

def stitch_out_of_core_results(
    tmp_out_dir: str, 
    resolution: float
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, csr_matrix, csr_matrix]:
    """
    Stitches disjoint count matrices, coordinates, and boundary edges from disk,
    resolving cross-boundary neighbors and returning stitched global matrices.
    """
    all_bins = []
    all_counts = []
    global_genes = None
    
    # Load all binned counts and active bin indexes
    for f in os.listdir(tmp_out_dir):
        if f.startswith("bins_") and f.endswith(".npy"):
            prefix = f[5:-4]
            bins = np.load(os.path.join(tmp_out_dir, f))
            counts = scipy.sparse.load_npz(os.path.join(tmp_out_dir, f"counts_{prefix}.npz"))
            genes = np.load(os.path.join(tmp_out_dir, f"genes_{prefix}.npy"), allow_pickle=True)
            
            all_bins.append(bins)
            all_counts.append(counts)
            if global_genes is None:
                global_genes = genes
                
    if not all_bins:
        raise ValueError("No intermediate results found to stitch.")
        
    # 1. Stitch global active bins list and expression matrix
    global_bins = np.concatenate(all_bins, axis=0)
    global_expression = scipy.sparse.vstack(all_counts).tocsr()
    
    offset_path = os.path.join(tmp_out_dir, "offset.npy")
    if os.path.exists(offset_path):
        x_min, y_min = np.load(offset_path)
        global_centers = np.column_stack((
            x_min + (global_bins[:, 0] + 0.5) * resolution,
            y_min + (global_bins[:, 1] + 0.5) * resolution
        ))
    else:
        global_centers = (global_bins + 0.5) * resolution
    
    # 2. Stitch and resolve duplicate graph edges
    all_edges = []
    for f in os.listdir(tmp_out_dir):
        if f.startswith("edges_") and f.endswith(".npy"):
            edges = np.load(os.path.join(tmp_out_dir, f))
            if len(edges) > 0:
                all_edges.append(edges)
                
    if all_edges:
        global_edges = np.concatenate(all_edges, axis=0)
        
        # Enforce canonical ordering: Node A < Node B lexicographically
        mask = (global_edges[:, 0] > global_edges[:, 2]) | (
            (global_edges[:, 0] == global_edges[:, 2]) & (global_edges[:, 1] > global_edges[:, 3])
        )
        temp = global_edges[mask, 0:2].copy()
        global_edges[mask, 0:2] = global_edges[mask, 2:4]
        global_edges[mask, 2:4] = temp
        
        # Remove duplicate cross-boundary edge list detections
        unique_edges = np.unique(global_edges, axis=0)
    else:
        unique_edges = np.empty((0, 4), dtype=np.int32)
        
    num_bins = len(global_bins)
    
    # 3. Map unique edges from grid coordinate keys to sequential indices in global_bins
    if len(unique_edges) > 0:
        gx_min = np.min(global_bins[:, 0])
        gy_min = np.min(global_bins[:, 1])
        # Use stride to create unique 1D key values
        stride = int(np.max(global_bins[:, 1]) - gy_min + 2)
        
        bin_keys = (global_bins[:, 0] - gx_min) * stride + (global_bins[:, 1] - gy_min)
        sort_idx = np.argsort(bin_keys)
        sorted_keys = bin_keys[sort_idx]
        
        edge_A_keys = (unique_edges[:, 0] - gx_min) * stride + (unique_edges[:, 1] - gy_min)
        edge_B_keys = (unique_edges[:, 2] - gx_min) * stride + (unique_edges[:, 3] - gy_min)
        
        idx_A = np.searchsorted(sorted_keys, edge_A_keys)
        idx_B = np.searchsorted(sorted_keys, edge_B_keys)
        
        row_indices = sort_idx[idx_A]
        col_indices = sort_idx[idx_B]
        
        # Assemble undirected symmetric adjacency matrix
        rows = np.concatenate([row_indices, col_indices])
        cols = np.concatenate([col_indices, row_indices])
        data = np.ones_like(rows, dtype=np.float32)
        
        adjacency_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(num_bins, num_bins),
            dtype=np.float32
        )
    else:
        adjacency_matrix = csr_matrix((num_bins, num_bins), dtype=np.float32)
        
    # Calculate Graph Laplacian: L = D - A
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    degree_matrix = csr_matrix(
        (degrees, (np.arange(num_bins), np.arange(num_bins))),
        shape=(num_bins, num_bins),
        dtype=np.float32
    )
    laplacian_matrix = degree_matrix - adjacency_matrix
    
    return global_expression, global_centers, global_genes, adjacency_matrix, laplacian_matrix
