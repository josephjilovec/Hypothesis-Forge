#!/usr/bin/env python3
"""
Main pipeline script for Hypothesis Forge.
Runs the complete workflow: preprocessing -> simulation -> hypothesis generation -> ranking.
"""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocess import preprocess_data
from simulation.sim_engine import SimulationEngine
from hypothesis.agent import HypothesisAgent
from hypothesis.hypothesis_ranking import HypothesisRanker
from knowledge.research_api import cross_reference_hypotheses
from knowledge.graph_builder import build_knowledge_graph
from utils.config_validator import validate_config
from utils.graceful_shutdown import get_shutdown_handler
import pandas as pd
import json

# Configure logging
try:
    from utils.logging_config import logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


def main():
    """Run the complete Hypothesis Forge pipeline."""
    # Validate configuration
    logger.info("Validating configuration...")
    validate_config()

    # Set up graceful shutdown
    shutdown_handler = get_shutdown_handler()

    def cleanup():
        logger.info("Cleaning up resources...")

    shutdown_handler.register_handler(cleanup)

    logger.info("=" * 60)
    logger.info("Starting Hypothesis Forge Pipeline")
    logger.info("=" * 60)

    try:
        # Step 1: Preprocess data
        logger.info("\n[1/5] Preprocessing data...")
        preprocess_data()
        logger.info("✓ Data preprocessing complete")

        # Step 2: Run simulations
        logger.info("\n[2/5] Running simulations...")
        engine = SimulationEngine()
        results_df = engine.run_simulations()
        logger.info(f"✓ Simulations complete: {len(results_df)} results")

        # Step 3: Generate hypotheses
        logger.info("\n[3/5] Generating hypotheses...")
        simulation_results = results_df.to_dict("records") if not results_df.empty else []
        agent = HypothesisAgent(simulation_results)
        agent.train(total_timesteps=5000)  # Reduced for faster execution
        hypotheses = agent.generate_hypotheses(num_hypotheses=20)
        logger.info(f"✓ Generated {len(hypotheses)} hypotheses")

        # Step 4: Rank hypotheses
        logger.info("\n[4/5] Ranking hypotheses...")
        ranker = HypothesisRanker()
        ranked_hypotheses = ranker.rank_hypotheses(hypotheses)
        logger.info(f"✓ Ranked {len(ranked_hypotheses)} hypotheses")

        # Step 5: Cross-reference with research APIs
        logger.info("\n[5/5] Cross-referencing hypotheses...")
        cross_referenced = cross_reference_hypotheses(ranked_hypotheses)
        logger.info(f"✓ Cross-referenced {len(cross_referenced)} hypotheses")

        # Optional: Build knowledge graph
        logger.info("\n[Optional] Building knowledge graph...")
        try:
            build_knowledge_graph(cross_referenced, simulation_results)
            logger.info("✓ Knowledge graph construction complete")
        except Exception as e:
            logger.warning(f"Knowledge graph construction skipped: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("Pipeline Complete!")
        logger.info("=" * 60)
        logger.info(f"Generated {len(cross_referenced)} ranked hypotheses")
        logger.info("View results in the Streamlit dashboard: streamlit run frontend/app.py")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

