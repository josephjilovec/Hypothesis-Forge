import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_simulation_results(results_dir: str = 'simulation/results') -> dict:
    """Load simulation results from parquet files."""
    try:
        results_dir = Path(results_dir)
        results = {}
        for file_path in results_dir.glob('*.parquet'):
            df = pd.read_parquet(file_path)
            sim_type = 'protein' if 'protein' in file_path.stem.lower() else 'astro'
            results[file_path.stem] = {'data': df, 'type': sim_type}
        logger.info(f"Loaded {len(results)} simulation results")
        return results
    except Exception as e:
        logger.error(f"Error loading simulation results: {str(e)}")
        return {}

def load_hypotheses(hypothesis_file: str = 'hypothesis/cross_referenced_hypotheses.json') -> list:
    """Load ranked hypotheses from JSON file."""
    try:
        hypothesis_file = Path(hypothesis_file)
        if hypothesis_file.exists():
            with open(hypothesis_file, 'r') as f:
                hypotheses = json.load(f)
            logger.info(f"Loaded {len(hypotheses)} hypotheses")
            return hypotheses
        logger.warning(f"Hypothesis file {hypothesis_file} not found")
        return []
    except Exception as e:
        logger.error(f"Error loading hypotheses: {str(e)}")
        return []

def plot_protein_structure(data: pd.DataFrame) -> go.Figure:
    """Create a 3D scatter plot for protein structure."""
    try:
        if data.shape[1] >= 3:
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    z=data.iloc[:, 2],
                    mode='markers+lines',
                    marker=dict(size=5, color='blue'),
                    line=dict(width=2, color='gray')
                )
            ])
            fig.update_layout(
                title="Protein Structure (C-alpha Coordinates)",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                showlegend=False
            )
            return fig
        return None
    except Exception as e:
        logger.error(f"Error plotting protein structure: {str(e)}")
        return None

def plot_astrophysical_data(data: pd.DataFrame) -> go.Figure:
    """Create a 3D scatter plot for astrophysical data."""
    try:
        if data.shape[1] >= 3:
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    z=data.iloc[:, 2],
                    mode='markers',
                    marker=dict(size=5, color=data.iloc[:, 3] if data.shape[1] > 3 else 'red', colorscale='Viridis')
                )
            ])
            fig.update_layout(
                title="Astrophysical System (Positions)",
                scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                showlegend=False
            )
            return fig
        return None
    except Exception as e:
        logger.error(f"Error plotting astrophysical data: {str(e)}")
        return None

def main():
    """Main function to create Streamlit dashboard."""
    st.set_page_config(page_title="Hypothesis Forge", layout="wide")
    
    # Load index.html template
    try:
        with open('frontend/templates/index.html', 'r') as f:
            html_template = f.read()
        logger.info("Loaded index.html template")
    except Exception as e:
        st.error(f"Error loading index.html: {str(e)}")
        logger.error(f"Error loading index.html: {str(e)}")
        return

    # Load data
    simulation_results = load_simulation_results()
    hypotheses = load_hypotheses()

    # Prepare data for JavaScript
    protein_data = []
    if simulation_results:
        for key, sim in simulation_results.items():
            if sim['type'] == 'protein':
                protein_data = sim['data'].values.tolist()[:100]  # Limit for performance
                break
    logger.info(f"Prepared protein_data with {len(protein_data)} rows")
    logger.info(f"Prepared hypotheses with {len(hypotheses)} items")

    # Sidebar for navigation
    st.sidebar.header("Navigation")
    view = st.sidebar.radio("Select View", ["Simulations", "Hypotheses"], key="view")

    # Render custom HTML with injected data
    try:
        components.html(
            html_template,
            height=800,
            scrolling=True,
            extra_args={
                'view': view,
                'proteinData': json.dumps(protein_data),
                'hypotheses': json.dumps(hypotheses)
            }
        )
        logger.info("Rendered HTML component with data")
    except Exception as e:
        st.error(f"Error rendering HTML component: {str(e)}")
        logger.error(f"Error rendering HTML component: {str(e)}")

    # Streamlit-native visualizations as fallback
    if view == "Simulations":
        st.header("Simulation Results")
        if not simulation_results:
            st.warning("No simulation results available.")
            return

        sim_key = st.selectbox("Select Simulation", list(simulation_results.keys()))
        sim_data = simulation_results[sim_key]['data']
        sim_type = simulation_results[sim_key]['type']

        if sim_type == 'protein':
            fig = plot_protein_structure(sim_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to generate protein structure plot.")
        else:
            fig = plot_astrophysical_data(sim_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to generate astrophysical plot.")

        with st.expander("View Raw Simulation Data"):
            st.dataframe(sim_data)

    elif view == "Hypotheses":
        st.header("Ranked Hypotheses")
        if not hypotheses:
            st.warning("No hypotheses available.")
            return

        df = pd.DataFrame(hypotheses)
        st.dataframe(
            df[['hypothesis', 'novelty', 'feasibility', 'score', 'novelty_api']],
            column_config={
                "hypothesis": "Hypothesis",
                "novelty": st.column_config.NumberColumn("Novelty (Sim)", format="%.3f"),
                "feasibility": st.column_config.NumberColumn("Feasibility", format="%.3f"),
                "score": st.column_config.NumberColumn("Score", format="%.3f"),
                "novelty_api": st.column_config.NumberColumn("Novelty (API)", format="%.3f")
            },
            use_container_width=True
        )

        fig = px.bar(
            df,
            x='hypothesis',
            y='score',
            title="Hypothesis Scores",
            labels={'hypothesis': 'Hypothesis', 'score': 'Score'},
            height=400
        )
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
