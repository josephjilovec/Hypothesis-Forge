"""
Neo4j knowledge graph builder.
Stores research relationships and computes novelty scores.
"""
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import json

from config.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from utils.logging_config import logger


class KnowledgeGraph:
    """Manages Neo4j knowledge graph for research relationships."""

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        """
        Initialize knowledge graph connection.

        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}. Running in offline mode.")
            self.driver = None

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")

    def create_hypothesis_node(self, hypothesis: Dict[str, Any]) -> Optional[str]:
        """
        Create a hypothesis node in the graph.

        Args:
            hypothesis: Hypothesis dictionary

        Returns:
            Node ID if successful, None otherwise
        """
        if not self.driver:
            logger.warning("Neo4j not connected, skipping node creation")
            return None

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    CREATE (h:Hypothesis {
                        id: $id,
                        text: $text,
                        type: $type,
                        novelty_score: $novelty_score,
                        composite_score: $composite_score,
                        created_at: datetime()
                    })
                    RETURN id(h) as node_id
                    """,
                    id=hypothesis.get("id", f"hyp_{hash(hypothesis.get('text', ''))}"),
                    text=hypothesis.get("text", ""),
                    type=hypothesis.get("type", 0),
                    novelty_score=hypothesis.get("novelty_score", 0.0),
                    composite_score=hypothesis.get("composite_score", 0.0),
                )
                record = result.single()
                node_id = record["node_id"] if record else None
                logger.info(f"Created hypothesis node: {node_id}")
                return str(node_id)
        except Exception as e:
            logger.error(f"Failed to create hypothesis node: {e}")
            return None

    def create_simulation_node(self, simulation: Dict[str, Any]) -> Optional[str]:
        """
        Create a simulation node in the graph.

        Args:
            simulation: Simulation result dictionary

        Returns:
            Node ID if successful, None otherwise
        """
        if not self.driver:
            logger.warning("Neo4j not connected, skipping node creation")
            return None

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    CREATE (s:Simulation {
                        id: $id,
                        type: $type,
                        success: $success,
                        num_steps: $num_steps,
                        runtime_seconds: $runtime_seconds,
                        created_at: datetime()
                    })
                    RETURN id(s) as node_id
                    """,
                    id=simulation.get("structure_id") or simulation.get("file_name", "unknown"),
                    type=simulation.get("simulation_type", "unknown"),
                    success=simulation.get("success", False),
                    num_steps=simulation.get("num_steps", 0),
                    runtime_seconds=simulation.get("runtime_seconds", 0.0),
                )
                record = result.single()
                node_id = record["node_id"] if record else None
                logger.info(f"Created simulation node: {node_id}")
                return str(node_id)
        except Exception as e:
            logger.error(f"Failed to create simulation node: {e}")
            return None

    def link_hypothesis_to_simulation(self, hypothesis_id: str, simulation_id: str):
        """
        Create relationship between hypothesis and simulation.

        Args:
            hypothesis_id: Hypothesis node ID
            simulation_id: Simulation node ID
        """
        if not self.driver:
            return

        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (h:Hypothesis {id: $hyp_id})
                    MATCH (s:Simulation {id: $sim_id})
                    CREATE (h)-[:GENERATED_FROM]->(s)
                    """,
                    hyp_id=hypothesis_id,
                    sim_id=simulation_id,
                )
                logger.info(f"Linked hypothesis {hypothesis_id} to simulation {simulation_id}")
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")

    def compute_novelty_score(self, hypothesis_text: str) -> float:
        """
        Compute novelty score by checking for similar hypotheses in graph.

        Args:
            hypothesis_text: Hypothesis text to check

        Returns:
            Novelty score (0-1, higher = more novel)
        """
        if not self.driver:
            # Return default score if Neo4j not available
            return 0.7

        try:
            with self.driver.session() as session:
                # Count similar hypotheses (simple keyword matching)
                result = session.run(
                    """
                    MATCH (h:Hypothesis)
                    WHERE h.text CONTAINS $keyword1 OR h.text CONTAINS $keyword2
                    RETURN count(h) as similar_count
                    """,
                    keyword1=hypothesis_text.split()[0] if hypothesis_text else "",
                    keyword2=hypothesis_text.split()[-1] if len(hypothesis_text.split()) > 1 else "",
                )
                record = result.single()
                similar_count = record["similar_count"] if record else 0

                # Novelty decreases with more similar hypotheses
                novelty = max(0.0, 1.0 - (similar_count * 0.1))
                return float(novelty)
        except Exception as e:
            logger.error(f"Failed to compute novelty score: {e}")
            return 0.7

    def get_related_hypotheses(self, hypothesis_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get hypotheses related to a given hypothesis.

        Args:
            hypothesis_id: Hypothesis node ID
            limit: Maximum number of related hypotheses to return

        Returns:
            List of related hypothesis dictionaries
        """
        if not self.driver:
            return []

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (h:Hypothesis {id: $id})-[:GENERATED_FROM]->(s:Simulation)<-[:GENERATED_FROM]-(related:Hypothesis)
                    WHERE h.id <> related.id
                    RETURN related
                    LIMIT $limit
                    """,
                    id=hypothesis_id,
                    limit=limit,
                )
                related = [dict(record["related"]) for record in result]
                return related
        except Exception as e:
            logger.error(f"Failed to get related hypotheses: {e}")
            return []


def build_knowledge_graph(hypotheses: List[Dict[str, Any]], simulations: List[Dict[str, Any]]):
    """
    Build knowledge graph from hypotheses and simulations.

    Args:
        hypotheses: List of hypothesis dictionaries
        simulations: List of simulation result dictionaries
    """
    logger.info("Building knowledge graph...")

    kg = KnowledgeGraph()
    kg.connect()

    if not kg.driver:
        logger.warning("Neo4j not available, knowledge graph operations skipped")
        return

    # Create simulation nodes
    sim_node_ids = {}
    for sim in simulations:
        sim_id = sim.get("structure_id") or sim.get("file_name", "unknown")
        node_id = kg.create_simulation_node(sim)
        if node_id:
            sim_node_ids[sim_id] = node_id

    # Create hypothesis nodes and link to simulations
    for hyp in hypotheses:
        hyp_id = hyp.get("id", f"hyp_{hash(hyp.get('text', ''))}")
        node_id = kg.create_hypothesis_node(hyp)

        # Link to simulation if context available
        context = hyp.get("simulation_context", {})
        sim_id = context.get("structure_id") or context.get("file_name")
        if sim_id and sim_id in sim_node_ids:
            kg.link_hypothesis_to_simulation(hyp_id, sim_node_ids[sim_id])

    kg.close()
    logger.info("Knowledge graph construction complete")


if __name__ == "__main__":
    # Example usage
    sample_hypotheses = [
        {
            "id": "hyp_1",
            "text": "Test hypothesis",
            "type": 0,
            "novelty_score": 0.8,
            "composite_score": 0.75,
            "simulation_context": {"structure_id": "test_protein"},
        }
    ]
    sample_simulations = [
        {
            "structure_id": "test_protein",
            "simulation_type": "protein_folding",
            "success": True,
            "num_steps": 50,
            "runtime_seconds": 120.0,
        }
    ]

    build_knowledge_graph(sample_hypotheses, sample_simulations)

