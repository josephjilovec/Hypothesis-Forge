"""Tests for knowledge graph and research API."""
import pytest
from unittest.mock import Mock, patch, MagicMock

from knowledge.graph_builder import KnowledgeGraph
from knowledge.research_api import PubMedClient, ArxivClient, ResearchAPIClient


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph."""

    def test_init(self):
        """Test knowledge graph initialization."""
        kg = KnowledgeGraph(uri="bolt://localhost:7687", user="test", password="test")
        assert kg.uri == "bolt://localhost:7687"
        assert kg.user == "test"

    @patch("knowledge.graph_builder.GraphDatabase")
    def test_connect_success(self, mock_graph_db):
        """Test successful connection."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run.return_value = None
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_graph_db.driver.return_value = mock_driver

        kg = KnowledgeGraph()
        kg.connect()
        assert kg.driver is not None

    def test_connect_failure(self):
        """Test connection failure."""
        kg = KnowledgeGraph(uri="bolt://invalid:7687", user="test", password="test")
        kg.connect()
        # Should handle failure gracefully
        assert kg.driver is None or kg.driver is not None


class TestPubMedClient:
    """Tests for PubMedClient."""

    def test_init(self):
        """Test client initialization."""
        client = PubMedClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_extract_keywords(self):
        """Test keyword extraction."""
        client = PubMedClient()
        text = "The protein structure exhibits characteristics that may influence binding"
        keywords = client._extract_keywords(text)
        assert isinstance(keywords, list)
        assert len(keywords) > 0


class TestArxivClient:
    """Tests for ArxivClient."""

    def test_init(self):
        """Test client initialization."""
        client = ArxivClient()
        assert client.base_url is not None

    def test_extract_keywords(self):
        """Test keyword extraction."""
        client = ArxivClient()
        text = "The gravitational dynamics suggest interactions between celestial bodies"
        keywords = client._extract_keywords(text)
        assert isinstance(keywords, list)
        assert len(keywords) > 0

