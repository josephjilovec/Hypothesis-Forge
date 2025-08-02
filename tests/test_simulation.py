import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
import time
import torch
from simulation.sim_engine import SimulationEngine, SimulationModel
from simulation.models.proteinsim import ProteinFoldingModel, SimulationModel as ProteinSim
from simulation.models.astrosim import GravitationalModel, SimulationModel as AstroSim
import logging

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def sim_engine(tmp_path):
    """Fixture to create a SimulationEngine instance with temporary directories."""
    data_dir = tmp_path / "data" / "processed"
    results_dir = tmp_path / "simulation" / "results"
    data_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    return SimulationEngine(data_dir=str(data_dir), output_dir=str(results_dir))

@pytest.fixture
def mock_data_file(tmp_path):
    """Fixture to create a mock parquet file with simulation data."""
    data_path = tmp_path / "data" / "processed" / "test_processed.parquet"
    data_path.parent.mkdir(parents=True)
    df = pd.DataFrame({
        'x': [1.0, 2.0, 3.0],
        'y': [4.0, 5.0, 6.0],
        'z': [7.0, 8.0, 9.0],
        'mass': [1e30, 1e30, 1e30]
    })
    df.to_parquet(data_path)
    return data_path

@pytest.fixture
def mock_protein_model():
    """Fixture to mock ProteinFoldingModel."""
    model = Mock(spec=ProteinFoldingModel)
    model.simulate.return_value = torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
    return model

@pytest.fixture
def mock_astro_model():
    """Fixture to mock GravitationalModel."""
    model = Mock(spec=GravitationalModel)
    model.simulate.return_value = torch.tensor([[1.2, 2.2, 3.2, 0.1, 0.2, 0.3]])
    return model

def test_simulation_engine_init(sim_engine, tmp_path):
    """Test SimulationEngine initialization."""
    assert sim_engine.data_dir == tmp_path / "data" / "processed"
    assert sim_engine.output_dir == tmp_path / "simulation" / "results"
    assert sim_engine.output_dir.exists()
    assert sim_engine.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_load_data_valid(sim_engine, mock_data_file):
    """Test loading valid parquet data."""
    df = sim_engine.load_data(mock_data_file)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 4)
    assert list(df.columns) == ['x', 'y', 'z', 'mass']
    assert df.notna().all().all()

def test_load_data_invalid(sim_engine, tmp_path):
    """Test loading invalid parquet file."""
    invalid_file = tmp_path / "data" / "processed" / "invalid.parquet"
    invalid_file.parent.mkdir(parents=True)
    invalid_file.write_text("INVALID DATA")
    
    with patch.object(logger, 'error') as mock_error:
        df = sim_engine.load_data(invalid_file)
        assert df is None
        mock_error.assert_called_once()
        assert "Error loading data" in mock_error.call_args[0][0]

def test_load_model_protein(sim_engine):
    """Test loading protein simulation model."""
    with patch('simulation.sim_engine.import_module') as mock_import:
        mock_module = Mock()
        mock_module.SimulationModel = ProteinSim
        mock_import.return_value = mock_module
        model = sim_engine.load_model('proteinsim')
        assert isinstance(model, ProteinSim)
        assert model.device == sim_engine.device

def test_load_model_astro(sim_engine):
    """Test loading astrophysical simulation model."""
    with patch('simulation.sim_engine.import_module') as mock_import:
        mock_module = Mock()
        mock_module.SimulationModel = AstroSim
        mock_import.return_value = mock_module
        model = sim_engine.load_model('astrosim')
        assert isinstance(model, AstroSim)
        assert model.device == sim_engine.device

def test_load_model_invalid(sim_engine):
    """Test loading invalid model."""
    with patch('simulation.sim_engine.import_module', side_effect=ImportError("Module not found")):
        with patch.object(logger, 'error') as mock_error:
            model = sim_engine.load_model('invalid_model')
            assert model is None
            mock_error.assert_called_once()
            assert "Error loading model" in mock_error.call_args[0][0]

def test_run_simulation_protein(sim_engine, mock_data_file, mock_protein_model):
    """Test running protein simulation."""
    data = pd.read_parquet(mock_data_file)
    start_time = time.time()
    results = sim_engine.run_simulation(data, mock_protein_model, max_duration=300)
    elapsed = time.time() - start_time
    assert results is not None
    assert isinstance(results, np.ndarray)
    assert results.shape == (2, 3)  # Based on mock_protein_model output
    assert elapsed < 300  # Ensure runtime < 5 minutes

def test_run_simulation_astro(sim_engine, mock_data_file, mock_astro_model):
    """Test running astrophysical simulation."""
    data = pd.read_parquet(mock_data_file)
    start_time = time.time()
    results = sim_engine.run_simulation(data, mock_astro_model, max_duration=300)
    elapsed = time.time() - start_time
    assert results is not None
    assert isinstance(results, np.ndarray)
    assert results.shape == (1, 6)  # Based on mock_astro_model output
    assert elapsed < 300  # Ensure runtime < 5 minutes

def test_run_simulation_timeout(sim_engine, mock_data_file, mock_protein_model):
    """Test simulation timeout handling."""
    with patch.object(mock_protein_model, 'simulate', side_effect=lambda x: time.sleep(301)):
        with patch.object(logger, 'error') as mock_error:
            results = sim_engine.run_simulation(pd.read_parquet(mock_data_file), mock_protein_model, max_duration=1)
            assert results is None
            mock_error.assert_called_once()
            assert "Simulation exceeded 1s" in mock_error.call_args[0][0]

def test_save_results(sim_engine, tmp_path):
    """Test saving simulation results."""
    results = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
    output_path = tmp_path / "simulation" / "results" / "test_results.parquet"
    
    with patch.object(logger, 'info') as mock_info:
        sim_engine.save_results(results, output_path)
        assert output_path.exists()
        saved_df = pd.read_parquet(output_path)
        assert saved_df.shape == (2, 3)
        np.testing.assert_array_equal(saved_df.values, results)
        mock_info.assert_called_once_with(f"Saved results to {output_path}")

def test_save_results_none(sim_engine, tmp_path):
    """Test saving None results."""
    output_path = tmp_path / "simulation" / "results" / "test_results.parquet"
    
    with patch.object(logger, 'warning') as mock_warning:
        sim_engine.save_results(None, output_path)
        assert not output_path.exists()
        mock_warning.assert_called_once_with(f"No results to save for {output_path}")

def test_protein_simulation_accuracy():
    """Test protein simulation output accuracy."""
    model = ProteinSim(device='cpu')
    data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    
    results = model.simulate(data)
    assert results is not None
    assert results.shape == (2, 3)
    # Check if results are close to input (simplified energy minimization)
    assert torch.allclose(results, data, rtol=0.1, atol=0.5)

def test_astro_simulation_accuracy():
    """Test astroph
