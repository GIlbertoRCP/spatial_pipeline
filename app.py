import streamlit as st
import pandas as pd
import numpy as np
import scipy.sparse
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO, StringIO
import os

from spatial_pipeline.binning import compute_square_bins
from spatial_pipeline.graphs import build_spatial_graph

# Configure Streamlit page options
st.set_page_config(
    page_title="Spatial Transcriptomics Preprocessing & Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown(
    """
    <style>
    .glass-header {
        background: rgba(255, 255, 255, 0.05);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 25px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 16px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="glass-header">
        <h1 style="color: #ECECF1; font-family: 'Outfit', sans-serif; font-weight: 700; margin: 0; font-size: 2.2rem;">Spatial Transcriptomics Preprocessing Dashboard</h1>
        <p style="color: #9EA0A3; font-family: 'Inter', sans-serif; font-size: 1.05rem; margin-top: 6px; margin-bottom: 0;">
            Interactive spatial grid binning, sparse matrix exporting, and Graph Laplacian visualizer using Streamlit & Plotly WebGL.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Cached generation of mock coordinate dataset
@st.cache_data
def get_mock_dataframe() -> pd.DataFrame:
    np.random.seed(42)
    n_molecules = 20000
    
    # Coordinates in 500x500 micron space
    x = np.random.uniform(50.0, 550.0, n_molecules)
    y = np.random.uniform(50.0, 550.0, n_molecules)
    
    # Create spatial clusters simulating cellular dense zones
    n_clusters = 8
    cluster_centers = np.random.uniform(120.0, 480.0, (n_clusters, 2))
    cluster_sizes = np.random.randint(800, 2000, n_clusters)
    
    for i in range(n_clusters):
        cx, cy = cluster_centers[i]
        size = cluster_sizes[i]
        c_noise_x = np.random.normal(0.0, 18.0, size)
        c_noise_y = np.random.normal(0.0, 18.0, size)
        x = np.concatenate([x, cx + c_noise_x])
        y = np.concatenate([y, cy + c_noise_y])
        
    x = np.clip(x, 50.0, 550.0)
    y = np.clip(y, 50.0, 550.0)
    
    # Assigned panel genes
    gene_panel = ["GAPDH", "ACTB", "EPCAM", "CD3D", "MS4A1", "ERBB2", "KRT19", "PECAM1", "COL1A1", "VIM"]
    gene_probs = np.random.dirichlet(np.ones(len(gene_panel)))
    assigned_genes = np.random.choice(gene_panel, size=len(x), p=gene_probs)
    
    return pd.DataFrame({
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "gene": pd.Series(assigned_genes, dtype="category")
    })

# Sidebar controls
st.sidebar.markdown("<h3 style='margin-bottom:0;'>Data Source Ingestion</h3>", unsafe_allow_html=True)
data_source = st.sidebar.radio(
    "Choose transcript dataset:",
    ["Use Mock Spatial Coordinates", "Upload Molecule Coordinates CSV"]
)

df = None
if data_source == "Use Mock Spatial Coordinates":
    df = get_mock_dataframe()
    st.sidebar.success(f"Loaded synthetic tissue sample ({len(df):,} molecules).")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload coordinates CSV file (columns: x, y, gene):",
        type=["csv", "gz"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(
                uploaded_file,
                usecols=["x", "y", "gene"],
                dtype={"x": "float32", "y": "float32", "gene": "category"}
            )
            st.sidebar.success(f"Ingested custom CSV ({len(df):,} molecules).")
        except Exception as e:
            st.sidebar.error(f"Error reading coordinate file: {e}")
    else:
        st.sidebar.info("Upload a molecule CSV to begin, or use the mock data option.")

if df is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3>Hyperparameters</h3>", unsafe_allow_html=True)
    
    resolution = st.sidebar.slider(
        "Square Grid Resolution (D):",
        min_value=2.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        help="The physical width of each grid block."
    )
    
    max_dist = st.sidebar.slider(
        "Adjacency Max Radius (T):",
        min_value=5.0,
        max_value=150.0,
        value=30.0,
        step=1.0,
        help="Max distance to connect two bins in the spatial graph."
    )
    
    # Execute Pipeline
    sparse_counts, bin_coords = compute_square_bins(df, resolution)
    adj_matrix, laplacian_matrix = build_spatial_graph(bin_coords, max_dist)
    
    # Compute Analytics Metrics
    n_molecules = len(df)
    n_bins = sparse_counts.shape[0]
    n_genes = sparse_counts.shape[1]
    n_edges = adj_matrix.nnz // 2
    sparsity = 100.0 * (1.0 - (sparse_counts.nnz / (n_bins * n_genes))) if (n_bins * n_genes) > 0 else 0.0
    
    # Render KPI Metric cards
    cols = st.columns(5)
    with cols[0]:
        st.markdown(
            f'<div class="metric-card"><div style="color: #9EA0A3; font-size: 0.85rem;">Molecules</div><div style="font-size: 1.6rem; font-weight: 700; color: #ECECF1;">{n_molecules:,}</div></div>', 
            unsafe_allow_html=True
        )
    with cols[1]:
        st.markdown(
            f'<div class="metric-card"><div style="color: #9EA0A3; font-size: 0.85rem;">Active Bins</div><div style="font-size: 1.6rem; font-weight: 700; color: #ECECF1;">{n_bins:,}</div></div>', 
            unsafe_allow_html=True
        )
    with cols[2]:
        st.markdown(
            f'<div class="metric-card"><div style="color: #9EA0A3; font-size: 0.85rem;">Genes</div><div style="font-size: 1.6rem; font-weight: 700; color: #ECECF1;">{n_genes}</div></div>', 
            unsafe_allow_html=True
        )
    with cols[3]:
        st.markdown(
            f'<div class="metric-card"><div style="color: #9EA0A3; font-size: 0.85rem;">Graph Edges</div><div style="font-size: 1.6rem; font-weight: 700; color: #ECECF1;">{n_edges:,}</div></div>', 
            unsafe_allow_html=True
        )
    with cols[4]:
        st.markdown(
            f'<div class="metric-card"><div style="color: #9EA0A3; font-size: 0.85rem;">Matrix Sparsity</div><div style="font-size: 1.6rem; font-weight: 700; color: #ECECF1;">{sparsity:.2f}%</div></div>', 
            unsafe_allow_html=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualization Tabs
    tab_molecules, tab_density, tab_graph, tab_exporter = st.tabs([
        "Molecules Map", 
        "Binned Expression Density", 
        "Spatial Network Overlay",
        "Exporter & Downloader"
    ])
    
    with tab_molecules:
        st.subheader("Interactive Transcript Detections Map")
        gene_options = ["All Genes"] + sorted(list(df["gene"].cat.categories))
        selected_gene = st.selectbox("Select gene to highlight:", gene_options)
        
        # Filter molecules
        if selected_gene == "All Genes":
            plot_df = df.sample(min(len(df), 30000), random_state=42)  # Cap plot size to prevent browser slowdown
            color_col = "gene"
            title_text = "Spatial transcripts map (Max 30k random molecules shown)"
        else:
            plot_df = df[df["gene"] == selected_gene]
            color_col = None
            title_text = f"Spatial transcripts map for gene: {selected_gene} ({len(plot_df):,} detections)"
            
        fig_mol = px.scatter(
            plot_df,
            x="x",
            y="y",
            color=color_col,
            title=title_text,
            template="plotly_dark",
            labels={"x": "X Position (um)", "y": "Y Position (um)"}
        )
        
        # Configure WebGL for fast hardware accelerated rendering
        fig_mol.update_traces(marker=dict(size=4, opacity=0.7), selector=dict(mode="markers"))
        fig_mol.update_layout(
            width=900,
            height=700,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117"
        )
        st.plotly_chart(fig_mol, use_container_width=True)
        
    with tab_density:
        st.subheader("Binned Molecule Densities")
        total_counts = np.array(sparse_counts.sum(axis=1)).flatten()
        
        # Plotly WebGL scatter plot for binned expression
        fig_density = go.Figure(
            data=go.Scattergl(
                x=bin_coords[:, 0],
                y=bin_coords[:, 1],
                mode="markers",
                marker=dict(
                    size=12,
                    color=total_counts,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Transcript Count"),
                    line=dict(width=0.5, color="rgba(255, 255, 255, 0.2)")
                ),
                text=[f"Bin {i}<br>Total Transcripts: {val}" for i, val in enumerate(total_counts)],
                hoverinfo="text+x+y"
            )
        )
        
        fig_density.update_layout(
            title=f"Binned expression density (Resolution={resolution})",
            template="plotly_dark",
            width=900,
            height=700,
            xaxis=dict(title="X position (um)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y position (um)"),
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117"
        )
        st.plotly_chart(fig_density, use_container_width=True)
        
    with tab_graph:
        st.subheader("Spatial Adjacency Graph Overlay")
        total_counts = np.array(sparse_counts.sum(axis=1)).flatten()
        
        fig_graph = go.Figure()
        
        # Extract edge list coordinates for fast drawing using singular trace
        if n_edges > 0:
            coo_adj = adj_matrix.tocoo()
            # Extract unique undirected pairs (rows < cols)
            mask = coo_adj.row < coo_adj.col
            edges_from = coo_adj.row[mask]
            edges_to = coo_adj.col[mask]
            
            # Format line segments separating with None values for performance
            edge_x = []
            edge_y = []
            for u, v in zip(edges_from, edges_to):
                edge_x.extend([bin_coords[u, 0], bin_coords[v, 0], None])
                edge_y.extend([bin_coords[u, 1], bin_coords[v, 1], None])
                
            fig_graph.add_trace(
                go.Scattergl(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(color="rgba(158, 160, 163, 0.35)", width=1),
                    hoverinfo="none",
                    name="Spatial Neighbors"
                )
            )
            
        # Draw node layers
        fig_graph.add_trace(
            go.Scattergl(
                x=bin_coords[:, 0],
                y=bin_coords[:, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color=total_counts,
                    colorscale="Inferno",
                    showscale=True,
                    colorbar=dict(title="Molecules/Bin"),
                    line=dict(width=0.3, color="rgba(255, 255, 255, 0.3)")
                ),
                text=[f"Bin Node {i}<br>Degree (Neighbors): {adj_matrix[i].sum()}" for i in range(n_bins)],
                hoverinfo="text+x+y",
                name="Bin Nodes"
            )
        )
        
        fig_graph.update_layout(
            title=f"Spatial Adjacency Graph (Max Distance={max_dist})",
            template="plotly_dark",
            width=900,
            height=700,
            xaxis=dict(title="X position (um)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y position (um)"),
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117"
        )
        st.plotly_chart(fig_graph, use_container_width=True)
        
    with tab_exporter:
        st.subheader("Export Generated Data Arrays")
        
        st.markdown(
            """
            This exporter allows you to convert the active binned counts, centers, and graph structures 
            into standard file formats for local analysis.
            """
        )
        
        # 1. Binned counts matrix csv
        st.markdown("##### 1. Binned Counts Matrix")
        # Build dense dataframe for simple download
        dense_counts_df = pd.DataFrame(
            sparse_counts.toarray(),
            columns=list(df["gene"].cat.categories)
        )
        dense_counts_df.insert(0, "bin_x", bin_coords[:, 0])
        dense_counts_df.insert(1, "bin_y", bin_coords[:, 1])
        
        csv_buffer = StringIO()
        dense_counts_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Binned Expression Matrix (CSV)",
            data=csv_buffer.getvalue(),
            file_name="expression_matrix.csv",
            mime="text/csv"
        )
        
        # 2. Graph Adjacency List
        st.markdown("##### 2. Spatial Graph Adjacency List")
        if n_edges > 0:
            coo_adj = adj_matrix.tocoo()
            adj_df = pd.DataFrame({
                "source_bin": coo_adj.row,
                "target_bin": coo_adj.col
            })
            # Remove duplicated undirected entries
            adj_df = adj_df[adj_df["source_bin"] < adj_df["target_bin"]]
            
            csv_adj_buffer = StringIO()
            adj_df.to_csv(csv_adj_buffer, index=False)
            st.download_button(
                label="Download Adjacency Edge List (CSV)",
                data=csv_adj_buffer.getvalue(),
                file_name="adjacency_edge_list.csv",
                mime="text/csv"
            )
        else:
            st.info("No connections between bins with current distance threshold. Increase Adjacency Max Radius.")
            
        # 3. Graph Laplacian Matrix (NPZ)
        st.markdown("##### 3. Graph Laplacian Sparse Matrix (NPZ)")
        npz_buffer = BytesIO()
        scipy.sparse.save_npz(npz_buffer, laplacian_matrix)
        st.download_button(
            label="Download Graph Laplacian Matrix (NPZ)",
            data=npz_buffer.getvalue(),
            file_name="laplacian_matrix.npz",
            mime="application/x-numpy"
        )
