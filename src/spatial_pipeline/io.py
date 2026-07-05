import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

def download_dataset(url: str, output_dir: str) -> str:
    """
    Downloads a public dataset from the given URL and extracts it if it is a zip file.
    
    Parameters
    ----------
    url : str
        The URL of the dataset to download.
    output_dir : str
        The directory where the downloaded file should be saved and extracted.
        
    Returns
    -------
    str
        The path to the directory containing the extracted files.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = url.split("/")[-1]
    download_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(download_path):
        print(f"Downloading dataset from {url}...")
        
        req = urllib.request.Request(
            url, 
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        
        with urllib.request.urlopen(req) as response, open(download_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 64  # 64 KB chunks
            count = 0
            
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                count += 1
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    percent = min(percent, 100)
                    print(f"Progress: {percent}% completed ({count * block_size}/{total_size} bytes)", end="\r")
                else:
                    print(f"Downloaded {count * block_size} bytes", end="\r")
                    
        print("\nDownload complete.")
    else:
        print(f"Dataset already exists at {download_path}")
        
    # Extract if zip
    if filename.endswith(".zip"):
        extract_dir = os.path.join(output_dir, filename[:-4])
        if not os.path.exists(extract_dir):
            print(f"Extracting {download_path} to {extract_dir}...")
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Extraction complete.")
        else:
            print(f"Files already extracted at {extract_dir}")
        return extract_dir
        
    return download_path

def load_coordinates(file_path: str, min_qv: float = 20.0) -> pd.DataFrame:
    """
    Loads spatial coordinates from a transcripts CSV, compressed CSV, or Parquet file.
    Attempts to use Polars for speed, falling back to Pandas.
    
    Parameters
    ----------
    file_path : str
        Path to the transcripts file.
    min_qv : float
        Minimum quality value threshold for transcripts. Transcripts with QV below
        this threshold will be discarded.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns: 'x', 'y', 'gene'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"Loading transcripts from {file_path}...")
    
    # 10x Genomics Xenium transcript column mappings
    column_mapping = {
        "x_location": "x",
        "y_location": "y",
        "feature_name": "gene",
        "x": "x",
        "y": "y",
        "gene": "gene",
        "qv": "qv"
    }
    
    if HAS_POLARS:
        print("Using Polars engine for fast coordinate loading.")
        # Load parquet or csv
        if file_path.endswith(".parquet"):
            df = pl.read_parquet(file_path)
        else:
            # Polars automatically detects gzip compression
            df = pl.read_csv(file_path)
            
        # Map columns
        present_cols = [c for c in column_mapping if c in df.columns]
        df = df.select(present_cols)
        rename_dict = {c: column_mapping[c] for c in present_cols}
        df = df.rename(rename_dict)
        
        # Filter by qv if present
        if "qv" in df.columns:
            df = df.filter(pl.col("qv") >= min_qv)
            df = df.drop("qv")
            
        # Convert to Pandas DataFrame for API standard
        pdf = df.to_pandas()
    else:
        print("Using Pandas engine for coordinate loading.")
        # Determine engine/format
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
            
        # Standardize columns
        present_cols = [c for c in column_mapping if c in df.columns]
        df = df[present_cols]
        df = df.rename(columns=column_mapping)
        
        # Filter by qv if present
        if "qv" in df.columns:
            df = df[df["qv"] >= min_qv]
            df = df.drop(columns=["qv"])
            
        pdf = df
        
    # Ensure correct types
    pdf["x"] = pdf["x"].astype("float64")
    pdf["y"] = pdf["y"].astype("float64")
    pdf["gene"] = pdf["gene"].astype("category")
    
    print(f"Loaded {len(pdf)} transcripts.")
    return pdf

def load_transcript_data(file_path: str) -> pd.DataFrame:
    """
    Loads and optimizes spatial transcriptomics coordinate files.
    Expects a CSV with columns: 'x', 'y', 'gene'
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No spatial transcript file found at {file_path}")
        
    # Read with explicit, low-memory datatypes to save RAM on big datasets
    df = pd.read_csv(
        path,
        usecols=['x', 'y', 'gene'],
        dtype={'x': 'float32', 'y': 'float32', 'gene': 'category'}
    )
    return df

def generate_vectorized_mock_data(file_path: str, num_molecules: int = 15000) -> None:
    """
    Instantly generates thousands of realistic spatial coordinates and mock gene labels
    using fully vectorized NumPy functions (no loops), then saves to a CSV.
    """
    print(f"[*] Instantly generating {num_molecules} synthetic transcripts using vectorized NumPy...")
    np.random.seed(42)
    
    # Half of the molecules are uniform background noise
    n_back = num_molecules // 2
    n_clust = num_molecules - n_back
    
    x_back = np.random.uniform(10.0, 500.0, n_back)
    y_back = np.random.uniform(10.0, 500.0, n_back)
    
    # Half are clustered around 5 spatial centers
    n_clusters = 5
    centers = np.random.uniform(50.0, 450.0, (n_clusters, 2))
    cluster_ids = np.random.randint(0, n_clusters, size=n_clust)
    assigned_centers = centers[cluster_ids]
    
    # Generate Gaussian noise around cluster centers vectorially
    noise = np.random.normal(0.0, 15.0, (n_clust, 2))
    x_clust = assigned_centers[:, 0] + noise[:, 0]
    y_clust = assigned_centers[:, 1] + noise[:, 1]
    
    # Concatenate
    x = np.concatenate([x_back, x_clust])
    y = np.concatenate([y_back, y_clust])
    
    # Clip coordinates to bounds
    x = np.clip(x, 10.0, 500.0)
    y = np.clip(y, 10.0, 500.0)
    
    # Vectorized gene assignment
    gene_list = ["GAPDH", "ACTB", "MALAT1", "EPCAM", "CD3D"]
    genes = np.random.choice(gene_list, size=num_molecules)
    
    df = pd.DataFrame({
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "gene": genes
    })
    
    df.to_csv(file_path, index=False)
    print(f"[+] Vectorized mock dataset saved to: {file_path}")
