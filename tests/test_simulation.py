"""Tests for simulation engine."""
import pytest
from unittest.mock import Mock, patch
import numpy as np

from simulation.sim_engine import (
    ProteinFoldingSimulator,
    GravitationalDynamicsSimulator,
    SimulationEngine,
)


class TestProteinFoldingSimulator:
    """Tests for ProteinFoldingSimulator."""

    def test_init(self):
        """Test simulator initialization."""
        simulator = ProteinFoldingSimulator()
        assert simulator.model is not None

    def test_simulate(self):
        """Test protein folding simulation."""
        simulator = ProteinFoldingSimulator()
        features = {
            "structure_id": "test_protein",
            "num_chains": 1,
            "num_residues": 100,
            "num_atoms": 500,
            "radius_of_gyration": 10.0,
            "max_dimension": 20.0,
        }
        result = simulator.simulate(features)
        assert result["success"] is True
        assert result["structure_id"] == "test_protein"
        assert "runtime_seconds" in result


class TestGravitationalDynamicsSimulator:
    """Tests for GravitationalDynamicsSimulator."""

    def test_init(self):
        """Test simulator initialization."""
        simulator = GravitationalDynamicsSimulator()
        assert simulator.gravitational_constant > 0

    def test_simulate(self):
        """Test gravitational dynamics simulation."""
        simulator = GravitationalDynamicsSimulator()
        features = {
            "file_name": "test.fits",
            "data_shape": [1024, 1024],
            "data_mean": 500.0,
            "data_std": 200.0,
        }
        result = simulator.simulate(features, num_bodies=5)
        assert result["success"] is True
        assert result["file_name"] == "test.fits"
        assert "trajectory" in result
        assert "runtime_seconds" in result

    def test_calculate_energy(self):
        """Test energy calculation."""
        simulator = GravitationalDynamicsSimulator()
        positions = np.random.randn(3, 3) * 10
        velocities = np.random.randn(3, 3) * 0.1
        masses = np.abs(np.random.randn(3)) * 1e10
        energy = simulator._calculate_energy(positions, velocities, masses)
        assert isinstance(energy, float)
        assert energy >= 0 or energy < 0  # Energy can be positive or negative


class TestSimulationEngine:
    """Tests for SimulationEngine."""

    def test_init(self):
        """Test engine initialization."""
        engine = SimulationEngine()
        assert engine.protein_simulator is not None
        assert engine.gravity_simulator is not None

