"""Tests for hypothesis generation and ranking."""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from hypothesis.agent import HypothesisGenerationEnv, HypothesisAgent
from hypothesis.hypothesis_ranking import HypothesisRanker


class TestHypothesisGenerationEnv:
    """Tests for HypothesisGenerationEnv."""

    def test_init(self):
        """Test environment initialization."""
        simulation_results = [
            {
                "simulation_type": "protein_folding",
                "num_steps": 50,
                "runtime_seconds": 120.0,
                "success": True,
            }
        ]
        env = HypothesisGenerationEnv(simulation_results)
        assert env.simulation_results == simulation_results
        assert env.action_space is not None
        assert env.observation_space is not None

    def test_reset(self):
        """Test environment reset."""
        simulation_results = [
            {
                "simulation_type": "protein_folding",
                "num_steps": 50,
                "runtime_seconds": 120.0,
                "success": True,
            }
        ]
        env = HypothesisGenerationEnv(simulation_results)
        obs, info = env.reset()
        assert obs.shape == (20,)
        assert "current_result" in info

    def test_step(self):
        """Test environment step."""
        simulation_results = [
            {
                "simulation_type": "protein_folding",
                "num_steps": 50,
                "runtime_seconds": 120.0,
                "success": True,
            }
        ]
        env = HypothesisGenerationEnv(simulation_results)
        obs, info = env.reset()
        action = np.array([0, 0.5, -0.5, 0.8])
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
        assert "hypothesis" in info


class TestHypothesisRanker:
    """Tests for HypothesisRanker."""

    def test_init(self):
        """Test ranker initialization."""
        ranker = HypothesisRanker()
        assert ranker.weights is not None
        assert "novelty" in ranker.weights

    def test_rank_hypotheses(self):
        """Test hypothesis ranking."""
        ranker = HypothesisRanker()
        hypotheses = [
            {
                "type": 0,
                "text": "Test hypothesis 1",
                "parameters": {"param1": 0.5, "param2": 0.3},
                "novelty_score": 0.8,
                "simulation_context": {"success": True, "num_steps": 50},
            },
            {
                "type": 1,
                "text": "Test hypothesis 2",
                "parameters": {"param1": 0.2, "param2": 0.1},
                "novelty_score": 0.6,
                "simulation_context": {"success": True, "num_steps": 30},
            },
        ]
        ranked = ranker.rank_hypotheses(hypotheses)
        assert len(ranked) == 2
        assert all("composite_score" in h for h in ranked)
        assert all("rank" in h for h in ranked)
        # Check that higher scores come first
        assert ranked[0]["composite_score"] >= ranked[1]["composite_score"]

    def test_calculate_coherence(self):
        """Test coherence calculation."""
        ranker = HypothesisRanker()
        hypothesis = {
            "parameters": {"param1": 0.2, "param2": 0.1},
        }
        coherence = ranker._calculate_coherence(hypothesis)
        assert 0.0 <= coherence <= 1.0

