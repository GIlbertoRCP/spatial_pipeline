import os
import zipfile
import urllib.request
import pandas as pd

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
