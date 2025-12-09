"""Tests for data preprocessing module."""
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from data.preprocess import ProteinPreprocessor, FITSPreprocessor


class TestProteinPreprocessor:
    """Tests for ProteinPreprocessor."""

    def test_init(self, tmp_path):
        """Test preprocessor initialization."""
        preprocessor = ProteinPreprocessor(tmp_path)
        assert preprocessor.pdb_dir == tmp_path
        assert preprocessor.parser is not None

    def test_extract_protein_features_empty_dir(self, tmp_path):
        """Test feature extraction with empty directory."""
        preprocessor = ProteinPreprocessor(tmp_path)
        result = preprocessor.process_all()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_process_all_no_files(self, tmp_path):
        """Test processing with no files."""
        preprocessor = ProteinPreprocessor(tmp_path)
        result = preprocessor.process_all()
        assert isinstance(result, pd.DataFrame)


class TestFITSPreprocessor:
    """Tests for FITSPreprocessor."""

    def test_init(self, tmp_path):
        """Test preprocessor initialization."""
        preprocessor = FITSPreprocessor(tmp_path)
        assert preprocessor.fits_dir == tmp_path

    def test_process_all_no_files(self, tmp_path):
        """Test processing with no files."""
        preprocessor = FITSPreprocessor(tmp_path)
        result = preprocessor.process_all()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


@pytest.fixture
def tmp_path():
    """Create temporary directory for tests."""
    import tempfile
    return Path(tempfile.mkdtemp())

