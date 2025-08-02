import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open, Mock
from data.preprocess import DataPreprocessor
import logging
from Bio.PDB import PDBParser
from astropy.io import fits

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def preprocessor(tmp_path):
    """Fixture to create a DataPreprocessor instance with temporary directories."""
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True)
    return DataPreprocessor(raw_dir=str(raw_dir), processed_dir=str(processed_dir))

@pytest.fixture
def mock_pdb_file(tmp_path):
    """Fixture to create a mock PDB file."""
    pdb_path = tmp_path / "data" / "raw" / "alphafold" / "test.pdb"
    pdb_path.parent.mkdir(parents=True)
    # Minimal valid PDB content
    pdb_content = """
ATOM      1  N   ALA A   1      10.000  20.000  30.000  1.00 20.00           N  
ATOM      2  CA  ALA A   1      11.000  21.000  31.000  1.00 20.00           C  
"""
    pdb_path.write_text(pdb_content)
    return pdb_path

@pytest.fixture
def mock_fits_file(tmp_path):
    """Fixture to create a mock FITS file."""
    fits_path = tmp_path / "data" / "raw" / "nasa" / "test.fits"
    fits_path.parent.mkdir(parents=True)
    # Create a simple FITS file with mock data
    hdu = fits.PrimaryHDU(data=np.array([[1.0, 2.0], [3.0, 4.0]]))
    hdu.writeto(fits_path)
    return fits_path

def test_init(preprocessor, tmp_path):
    """Test DataPreprocessor initialization."""
    assert preprocessor.raw_dir == tmp_path / "data" / "raw"
    assert preprocessor.processed_dir == tmp_path / "data" / "processed"
    assert preprocessor.processed_dir.exists()

def test_process_pdb_file_valid(preprocessor, mock_pdb_file):
    """Test processing of a valid PDB file."""
    df = preprocessor.process_pdb_file(mock_pdb_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'sequence' in df.columns
    assert 'coords' in df.columns
    assert len(df) > 0
    assert df['sequence'].notna().all()

def test_process_pdb_file_invalid(preprocessor, tmp_path):
    """Test processing of an invalid PDB file."""
    invalid_pdb = tmp_path / "data" / "raw" / "alphafold" / "invalid.pdb"
    invalid_pdb.parent.mkdir(parents=True)
    invalid_pdb.write_text("INVALID DATA")
    
    with patch.object(logger, 'error') as mock_error:
        df = preprocessor.process_pdb_file(invalid_pdb)
        assert df is None
        mock_error.assert_called_once()
        assert "Error processing PDB file" in mock_error.call_args[0][0]

def test_process_fits_file_valid(preprocessor, mock_fits_file):
    """Test processing of a valid FITS file."""
    df = preprocessor.process_fits_file(mock_fits_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[1] == 2  # Based on mock data
    assert df.notna().all().all()  # No missing values after filling
    # Check normalization
    assert np.allclose(df.mean(), 0, atol=1e-5)
    assert np.allclose(df.std(), 1, atol=1e-5)

def test_process_fits_file_invalid(preprocessor, tmp_path):
    """Test processing of an invalid FITS file."""
    invalid_fits = tmp_path / "data" / "raw" / "nasa" / "invalid.fits"
    invalid_fits.parent.mkdir(parents=True)
    invalid_fits.write_text("INVALID DATA")
    
    with patch.object(logger, 'error') as mock_error:
        df = preprocessor.process_fits_file(invalid_fits)
        assert df is None
        mock_error.assert_called_once()
        assert "Error processing FITS file" in mock_error.call_args[0][0]

def test_process_dataset_pdb(preprocessor, mock_pdb_file):
    """Test process_dataset dispatching for PDB files."""
    with patch('data.preprocess.DataPreprocessor.process_pdb_file') as mock_pdb:
        mock_pdb.return_value = pd.DataFrame({'sequence': ['ALA'], 'coords': [[[1, 2, 3]]]})
        df = preprocessor.process_dataset(mock_pdb_file)
        assert isinstance(df, pd.DataFrame)
        mock_pdb.assert_called_once_with(mock_pdb_file)

def test_process_dataset_fits(preprocessor, mock_fits_file):
    """Test process_dataset dispatching for FITS files."""
    with patch('data.preprocess.DataPreprocessor.process_fits_file') as mock_fits:
        mock_fits.return_value = pd.DataFrame([[1, 2]])
        df = preprocessor.process_dataset(mock_fits_file)
        assert isinstance(df, pd.DataFrame)
        mock_fits.assert_called_once_with(mock_fits_file)

def test_process_dataset_unsupported(preprocessor, tmp_path):
    """Test process_dataset with unsupported file format."""
    unsupported_file = tmp_path / "data" / "raw" / "test.txt"
    unsupported_file.parent.mkdir(parents=True)
    unsupported_file.write_text("test")
    
    with patch.object(logger, 'error') as mock_error:
        df = preprocessor.process_dataset(unsupported_file)
        assert df is None
        mock_error.assert_called_once()
        assert "Unsupported file format: .txt" in mock_error.call_args[0][0]

def test_save_processed_data(preprocessor, tmp_path):
    """Test saving processed data to parquet."""
    df = pd.DataFrame({'sequence': ['ALA'], 'coords': [[[1, 2, 3]]]})
    output_path = tmp_path / "data" / "processed" / "test_processed.parquet"
    
    with patch.object(logger, 'info') as mock_info:
        preprocessor.save_processed_data(df, output_path)
        assert output_path.exists()
        saved_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(saved_df, df)
        mock_info.assert_called_once_with(f"Saved processed data to {output_path}")

def test_save_processed_data_error(preprocessor, tmp_path):
    """Test error handling in save_processed_data."""
    df = pd.DataFrame({'sequence': ['ALA'], 'coords': [[[1, 2, 3]]]})
    output_path = tmp_path / "data" / "processed" / "invalid/invalid.parquet"
    
    with patch('pandas.DataFrame.to_parquet', side_effect=Exception("Write error")):
        with patch.object(logger, 'error') as mock_error:
            preprocessor.save_processed_data(df, output_path)
            mock_error.assert_called_once()
            assert f"Error saving processed data to {output_path}" in mock_error.call_args[0][0]

def test_run(preprocessor, mock_pdb_file, mock_fits_file):
    """Test full run method with valid files."""
    with patch('data.preprocess.DataPreprocessor.process_dataset') as mock_process:
        mock_process.side_effect = [
            pd.DataFrame({'sequence': ['ALA'], 'coords': [[[1, 2, 3]]]}),
            pd.DataFrame([[1, 2]])
        ]
        with patch('data.preprocess.DataPreprocessor.save_processed_data') as mock_save:
            preprocessor.run()
            assert mock_process.call_count == 2
            assert mock_save.call_count == 2

def test_run_empty_directory(preprocessor, tmp_path):
    """Test run method with empty raw directory."""
    # Clear raw directory
    (tmp_path / "data" / "raw" / "alphafold").rmdir()
    (tmp_path / "data" / "raw" / "nasa").rmdir()
    
    with patch.object(logger, 'warning') as mock_warning:
        preprocessor.run()
        assert mock_warning.call_count == 2
        assert "Directory" in mock_warning.call_args_list[0][0][0]
        assert "Directory" in mock_warning.call_args_list[1][0][0]

if __name__ == "__main__":
    pytest.main(["-v"])
