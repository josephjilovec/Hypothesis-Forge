"""
Streamlit dashboard for Hypothesis Forge.
Displays simulation results and ranked hypotheses with interactive visualizations.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import logging

from config.config import RESULTS_DIR, HYPOTHESIS_DIR, PROCESSED_DATA_DIR

# Configure logging
try:
    from utils.logging_config import logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Hypothesis Forge",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_simulation_results():
    """Load simulation results from parquet file."""
    results_file = RESULTS_DIR / "simulation_results.parquet"
    if results_file.exists():
        try:
            return pd.read_parquet(results_file)
        except Exception as e:
            logger.error(f"Failed to load simulation results: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data
def load_hypotheses():
    """Load ranked hypotheses from JSON file."""
    hypotheses_file = HYPOTHESIS_DIR / "ranked_hypotheses.json"
    if hypotheses_file.exists():
        try:
            with open(hypotheses_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load hypotheses: {e}")
            return []
    return []


@st.cache_data
def load_cross_referenced_hypotheses():
    """Load cross-referenced hypotheses."""
    cross_ref_file = HYPOTHESIS_DIR / "cross_referenced_hypotheses.json"
    if cross_ref_file.exists():
        try:
            with open(cross_ref_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cross-referenced hypotheses: {e}")
            return []
    return []


def render_simulations_page():
    """Render simulations visualization page."""
    st.header("🔬 Simulation Results")

    results_df = load_simulation_results()

    if results_df.empty:
        st.warning("No simulation results available. Run simulations first.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Simulations", len(results_df))
    with col2:
        success_count = results_df["success"].sum() if "success" in results_df.columns else 0
        st.metric("Successful", success_count)
    with col3:
        avg_runtime = results_df["runtime_seconds"].mean() if "runtime_seconds" in results_df.columns else 0
        st.metric("Avg Runtime (s)", f"{avg_runtime:.2f}")
    with col4:
        total_steps = results_df["num_steps"].sum() if "num_steps" in results_df.columns else 0
        st.metric("Total Steps", total_steps)

    # Simulation type distribution
    if "simulation_type" in results_df.columns:
        st.subheader("Simulation Type Distribution")
        type_counts = results_df["simulation_type"].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Simulation Types",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Runtime distribution
    if "runtime_seconds" in results_df.columns:
        st.subheader("Runtime Distribution")
        fig = px.histogram(
            results_df,
            x="runtime_seconds",
            nbins=20,
            title="Simulation Runtime Distribution",
            labels={"runtime_seconds": "Runtime (seconds)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Success rate by type
    if "simulation_type" in results_df.columns and "success" in results_df.columns:
        st.subheader("Success Rate by Simulation Type")
        success_by_type = results_df.groupby("simulation_type")["success"].mean() * 100
        fig = px.bar(
            x=success_by_type.index,
            y=success_by_type.values,
            title="Success Rate (%) by Simulation Type",
            labels={"x": "Simulation Type", "y": "Success Rate (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed results table
    st.subheader("Detailed Results")
    st.dataframe(results_df, use_container_width=True)


def render_hypotheses_page():
    """Render hypotheses visualization page."""
    st.header("💡 Ranked Hypotheses")

    hypotheses = load_hypotheses()

    if not hypotheses:
        st.warning("No hypotheses available. Generate hypotheses first.")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Hypotheses", len(hypotheses))
    with col2:
        avg_novelty = sum(h.get("novelty_score", 0) for h in hypotheses) / len(hypotheses) if hypotheses else 0
        st.metric("Avg Novelty Score", f"{avg_novelty:.2f}")
    with col3:
        avg_composite = sum(h.get("composite_score", 0) for h in hypotheses) / len(hypotheses) if hypotheses else 0
        st.metric("Avg Composite Score", f"{avg_composite:.2f}")

    # Top hypotheses
    st.subheader("Top Ranked Hypotheses")
    top_n = st.slider("Number of top hypotheses to display", 1, min(20, len(hypotheses)), 10)

    for i, hypothesis in enumerate(hypotheses[:top_n]):
        with st.expander(f"Rank {hypothesis.get('rank', i+1)}: {hypothesis.get('text', 'No text')[:100]}..."):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Novelty Score", f"{hypothesis.get('novelty_score', 0):.3f}")
                st.metric("Composite Score", f"{hypothesis.get('composite_score', 0):.3f}")
            with col2:
                st.metric("Type", hypothesis.get('type', 'Unknown'))
                st.metric("Rank", hypothesis.get('rank', i+1))

            st.write("**Full Text:**")
            st.write(hypothesis.get('text', 'No text available'))

            if "parameters" in hypothesis:
                st.write("**Parameters:**")
                st.json(hypothesis["parameters"])

    # Novelty score distribution
    st.subheader("Novelty Score Distribution")
    novelty_scores = [h.get("novelty_score", 0) for h in hypotheses]
    fig = px.histogram(
        x=novelty_scores,
        nbins=20,
        title="Novelty Score Distribution",
        labels={"x": "Novelty Score", "y": "Count"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Composite score vs novelty
    st.subheader("Composite Score vs Novelty")
    composite_scores = [h.get("composite_score", 0) for h in hypotheses]
    fig = px.scatter(
        x=novelty_scores,
        y=composite_scores,
        title="Composite Score vs Novelty Score",
        labels={"x": "Novelty Score", "y": "Composite Score"},
        hover_data=[list(range(len(hypotheses)))],
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cross_reference_page():
    """Render cross-referenced hypotheses page."""
    st.header("📚 Cross-Referenced Hypotheses")

    hypotheses = load_cross_referenced_hypotheses()

    if not hypotheses:
        st.warning("No cross-referenced hypotheses available.")
        return

    # Summary
    st.metric("Cross-Referenced Hypotheses", len(hypotheses))

    # Display hypotheses with cross-reference data
    for i, hypothesis in enumerate(hypotheses):
        with st.expander(f"Hypothesis {i+1}: {hypothesis.get('text', 'No text')[:80]}..."):
            st.write("**Hypothesis Text:**")
            st.write(hypothesis.get('text', 'No text available'))

            if "cross_reference" in hypothesis:
                cross_ref = hypothesis["cross_reference"]
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("PubMed")
                    pubmed = cross_ref.get("pubmed", {})
                    st.metric("Similar Articles", pubmed.get("similar_articles_count", 0))
                    st.metric("Novelty Score", f"{pubmed.get('novelty_score', 0):.3f}")

                with col2:
                    st.subheader("arXiv")
                    arxiv = cross_ref.get("arxiv", {})
                    st.metric("Similar Papers", arxiv.get("similar_papers_count", 0))
                    st.metric("Novelty Score", f"{arxiv.get('novelty_score', 0):.3f}")

                st.metric(
                    "Combined Novelty",
                    f"{cross_ref.get('combined_novelty_score', 0):.3f}",
                )


def main():
    """Main dashboard application."""
    st.markdown('<p class="main-header">🔬 Hypothesis Forge</p>', unsafe_allow_html=True)
    st.markdown("**AI-driven scientific hypothesis generation and ranking system**")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Simulations", "Hypotheses", "Cross-References"],
    )

    # Render selected page
    if page == "Simulations":
        render_simulations_page()
    elif page == "Hypotheses":
        render_hypotheses_page()
    elif page == "Cross-References":
        render_cross_reference_page()

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        """
        Hypothesis Forge is an AI-driven system for generating
        and ranking novel scientific hypotheses using reinforcement
        learning and knowledge graphs.
        """
    )


if __name__ == "__main__":
    main()

