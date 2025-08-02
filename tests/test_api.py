import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock
import logging
from knowledge.research_api import ResearchAPI
import pandas as pd
import numpy as np

# Set up logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def research_api(tmp_path):
    """Fixture to create a ResearchAPI instance with temporary directories."""
    hypothesis_file = tmp_path / "hypothesis" / "cross_referenced_hypotheses.json"
    hypothesis_file.parent.mkdir(parents=True)
    return ResearchAPI(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        db_name="neo4j",
        hypothesis_file=str(hypothesis_file)
    )

@pytest.fixture
def mock_hypotheses(tmp_path):
    """Fixture to create mock hypotheses JSON file."""
    hypothesis_file = tmp_path / "hypothesis" / "cross_referenced_hypotheses.json"
    hypotheses = [
        {"hypothesis": "Increase temperature affects protein stability", "state": [1.0] * 10},
        {"hypothesis": "Alter gravitational field changes orbital dynamics", "state": [2.0] * 10}
    ]
    hypothesis_file.parent.mkdir(parents=True)
    with open(hypothesis_file, 'w') as f:
        json.dump(hypotheses, f)
    return hypotheses

@pytest.fixture
def mock_neo4j_driver():
    """Fixture to mock Neo4j driver."""
    driver = Mock()
    session = Mock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__ = Mock()
    return driver

def test_research_api_init(research_api, tmp_path):
    """Test ResearchAPI initialization."""
    assert research_api.hypothesis_file == tmp_path / "hypothesis" / "cross_referenced_hypotheses.json"
    assert research_api.uri == "bolt://localhost:7687"
    assert research_api.user == "neo4j"
    assert research_api.password == "password"
    assert research_api.db_name == "neo4j"

def test_connect_success(research_api, mock_neo4j_driver):
    """Test successful Neo4j connection."""
    with patch('knowledge.research_api.GraphDatabase.driver', return_value=mock_neo4j_driver):
        with patch.object(logger, 'info') as mock_info:
            research_api.connect()
            assert research_api.driver == mock_neo4j_driver
            mock_info.assert_called_once_with(f"Connected to Neo4j at {research_api.uri}")

def test_connect_failure(research_api):
    """Test Neo4j connection failure."""
    with patch('knowledge.research_api.GraphDatabase.driver', side_effect=Exception("Connection error")):
        with patch.object(logger, 'error') as mock_error:
            research_api.connect()
            assert research_api.driver is None
            mock_error.assert_called_once()
            assert "Failed to connect to Neo4j" in mock_error.call_args[0][0]

def test_close(research_api, mock_neo4j_driver):
    """Test closing Neo4j connection."""
    research_api.driver = mock_neo4j_driver
    with patch.object(logger, 'info') as mock_info:
        research_api.close()
        mock_neo4j_driver.close.assert_called_once()
        mock_info.assert_called_once_with("Closed Neo4j connection")

def test_load_hypotheses(research_api, mock_hypotheses, tmp_path):
    """Test loading hypotheses from JSON file."""
    hypotheses = research_api.load_hypotheses()
    assert len(hypotheses) == 2
    assert hypotheses[0]['hypothesis'] == "Increase temperature affects protein stability"
    assert len(hypotheses[0]['state']) == 10

def test_load_hypotheses_missing_file(research_api, tmp_path):
    """Test loading hypotheses when file is missing."""
    research_api.hypothesis_file = tmp_path / "hypothesis" / "nonexistent.json"
    with patch.object(logger, 'warning') as mock_warning:
        hypotheses = research_api.load_hypotheses()
        assert hypotheses == []
        mock_warning.assert_called_once()
        assert "not found" in mock_warning.call_args[0][0]

def test_query_pubmed_success(research_api):
    """Test successful PubMed API query."""
    mock_response = Mock()
    mock_response.json.return_value = {
        'esearchresult': {'idlist': ['12345', '67890']}
    }
    mock_summary_response = Mock()
    mock_summary_response.json.return_value = {
        'result': {
            '12345': {'title': 'Test Paper 1', 'keywords': ['protein', 'stability']},
            '67890': {'title': 'Test Paper 2', 'keywords': ['temperature']}
        }
    }
    with patch('requests.get', side_effect=[mock_response, mock_summary_response, mock_summary_response]) as mock_get:
        papers = research_api.query_pubmed("protein stability", max_results=2)
        assert len(papers) == 2
        assert papers[0]['id'] == '12345'
        assert papers[0]['title'] == 'Test Paper 1'
        assert papers[0]['source'] == 'PubMed'
        assert 'protein' in papers[0]['topics']
        assert mock_get.call_count == 3  # esearch + 2 esummary calls

def test_query_pubmed_failure(research_api):
    """Test PubMed API query failure."""
    with patch('requests.get', side_effect=Exception("API error")):
        with patch.object(logger, 'error') as mock_error:
            papers = research_api.query_pubmed("protein stability")
            assert papers == []
            mock_error.assert_called_once()
            assert "Error querying PubMed" in mock_error.call_args[0][0]

def test_query_arxiv_success(research_api):
    """Test successful arXiv API query."""
    mock_response = Mock()
    mock_response.content = """
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>http://arxiv.org/abs/1234.5678</id>
            <title>Test Paper</title>
            <category term="astro-ph"/>
        </entry>
    </feed>
    """
    with patch('requests.get', return_value=mock_response):
        papers = research_api.query_arxiv("gravitational field", max_results=1)
        assert len(papers) == 1
        assert papers[0]['id'] == '1234.5678'
        assert papers[0]['title'] == 'Test Paper'
        assert papers[0]['source'] == 'arXiv'
        assert 'astro-ph' in papers[0]['topics']

def test_query_arxiv_failure(research_api):
    """Test arXiv API query failure."""
    with patch('requests.get', side_effect=Exception("API error")):
        with patch.object(logger, 'error') as mock_error:
            papers = research_api.query_arxiv("gravitational field")
            assert papers == []
            mock_error.assert_called_once()
            assert "Error querying arXiv" in mock_error.call_args[0][0]

def test_update_graph(research_api, mock_neo4j_driver):
    """Test updating Neo4j graph with hypothesis and papers."""
    research_api.driver = mock_neo4j_driver
    papers = [
        {'id': '12345', 'title': 'Test Paper', 'source': 'PubMed'},
        {'id': '67890', 'title': 'Another Paper', 'source': 'arXiv'}
    ]
    with patch.object(mock_neo4j_driver.session().__enter__(), 'run') as mock_run:
        research_api.update_graph("Test hypothesis", papers)
        assert mock_run.call_count == 3  # Hypothesis node + 2 paper relationships
        mock_run.assert_any_call(
            "MERGE (h:Hypothesis {text: $hypothesis})",
            hypothesis="Test hypothesis"
        )

def test_compute_novelty(research_api, mock_neo4j_driver):
    """Test computing novelty score."""
    research_api.driver = mock_neo4j_driver
    mock_result = Mock()
    mock_result.single.return_value = {'paper_count': 2}
    mock_neo4j_driver.session().__enter__().run.return_value = mock_result
    
    novelty = research_api.compute_novelty("Test hypothesis")
    assert isinstance(novelty, float)
    assert novelty == pytest.approx(1.0 / (1.0 + 2), 1e-5)  # 1 / (1 + paper_count)

def test_compute_novelty_no_driver(research_api):
    """Test novelty computation without Neo4j driver."""
    research_api.driver = None
    with patch.object(logger, 'error') as mock_error:
        novelty = research_api.compute_novelty("Test hypothesis")
        assert novelty == 0.5
        mock_error.assert_called_once()
        assert "No Neo4j connection available" in mock_error.call_args[0][0]

def test_save_results(research_api, tmp_path, mock_hypotheses):
    """Test saving cross-referenced hypotheses."""
    output_file = tmp_path / "hypothesis" / "cross_referenced_hypotheses.json"
    research_api.save_results(mock_hypotheses)
    assert output_file.exists()
    with open(output_file, 'r') as f:
        saved_data = json.load(f)
    assert len(saved_data) == 2
    assert saved_data[0]['hypothesis'] == mock_hypotheses[0]['hypothesis']

def test_cross_reference_hypotheses(research_api, mock_hypotheses, mock_neo4j_driver):
    """Test cross-referencing hypotheses with APIs and Neo4j."""
    research_api.driver = mock_neo4j_driver
    mock_pubmed = Mock(return_value=[{'id': '12345', 'title': 'Test Paper', 'source': 'PubMed'}])
    mock_arxiv = Mock(return_value=[{'id': '67890', 'title': 'Another Paper', 'source': 'arXiv'}])
    mock_novelty = Mock(return_value=0.8)
    
    with patch.multiple(research_api,
                        query_pubmed=mock_pubmed,
                        query_arxiv=mock_arxiv,
                        update_graph=Mock(),
                        compute_novelty=mock_novelty):
        with patch('knowledge.research_api.HypothesisRanker.save_ranked_hypotheses') as mock_save:
            results = research_api.cross_reference_hypotheses()
            assert len(results) == 2
            assert results[0]['novelty_api'] == 0.8
            assert mock_pubmed.call_count == 2
            assert mock_arxiv.call_count == 2
            assert mock_novelty.call_count == 2
            assert mock_save.called

def test_run_integration(research_api, mock_hypotheses):
    """Test full run method."""
    with patch.object(research_api, 'cross_reference_hypotheses', return_value=mock_hypotheses) as mock_cross:
        result = research_api.run()
        assert result == mock_hypotheses
        mock_cross.assert_called_once()

if __name__ == "__main__":
    pytest.main(["-v"])
