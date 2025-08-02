import json
import requests
import xml.etree.ElementTree as ET
from neo4j import GraphDatabase
from pathlib import Path
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResearchAPI:
    """Queries PubMed and arXiv APIs to check hypothesis novelty using Neo4j knowledge graph."""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password", db_name: str = "neo4j",
                 hypothesis_file: str = 'hypothesis/ranked_hypotheses.json'):
        self.hypothesis_file = Path(hypothesis_file)
        self.driver = None
        self.uri = neo4j_uri
        self.user = neo4j_user
        self.password = neo4j_password
        self.db_name = db_name
        self.connect()
        logger.info("Initialized ResearchAPI")

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            self.driver = None

    def close(self) -> None:
        """Close Neo4j connection."""
        try:
            if self.driver:
                self.driver.close()
                logger.info("Closed Neo4j connection")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")

    def load_hypotheses(self) -> List[Dict]:
        """Load ranked hypotheses from JSON file."""
        try:
            if self.hypothesis_file.exists():
                with open(self.hypothesis_file, 'r') as f:
                    return json.load(f)
            logger.warning(f"No hypotheses found at {self.hypothesis_file}")
            return []
        except Exception as e:
            logger.error(f"Error loading hypotheses from {self.hypothesis_file}: {str(e)}")
            return []

    def query_pubmed(self, hypothesis: str, max_results: int = 5) -> List[Dict]:
        """Query PubMed API for papers related to hypothesis."""
        try:
            query = hypothesis.replace(" ", "+")
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            papers = []
            ids = data.get('esearchresult', {}).get('idlist', [])
            for pubmed_id in ids:
                summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pubmed_id}&retmode=json"
                summary_response = requests.get(summary_url, timeout=10)
                summary_response.raise_for_status()
                summary = summary_response.json()
                
                doc = summary.get('result', {}).get(pubmed_id, {})
                papers.append({
                    'id': pubmed_id,
                    'title': doc.get('title', ''),
                    'source': 'PubMed'
                })
            logger.info(f"Fetched {len(papers)} PubMed papers for hypothesis: {hypothesis}")
            return papers
        except Exception as e:
            logger.error(f"Error querying PubMed for hypothesis {hypothesis}: {str(e)}")
            return []

    def query_arxiv(self, hypothesis: str, max_results: int = 5) -> List[Dict]:
        """Query arXiv API for papers related to hypothesis."""
        try:
            query = hypothesis.replace(" ", "+")
            url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                title = entry.find('{http://www.w3.org/2005/Atom}title').text
                papers.append({
                    'id': arxiv_id,
                    'title': title,
                    'source': 'arXiv'
                })
            logger.info(f"Fetched {len(papers)} arXiv papers for hypothesis: {hypothesis}")
            return papers
        except Exception as e:
            logger.error(f"Error querying arXiv for hypothesis {hypothesis}: {str(e)}")
            return []

    def update_graph(self, hypothesis: str, papers: List[Dict]) -> None:
        """Update Neo4j graph with hypothesis and related papers."""
        try:
            if not self.driver:
                logger.error("No Neo4j connection available")
                return
            
            with self.driver.session(database=self.db_name) as session:
                # Create Hypothesis node
                session.run(
                    """
                    MERGE (h:Hypothesis {text: $hypothesis})
                    """,
                    hypothesis=hypothesis
                )
                
                # Create Paper nodes and relationships
                for paper in papers:
                    session.run(
                        """
                        MERGE (p:Paper {id: $id, source: $source})
                        SET p.title = $title
                        MERGE (h:Hypothesis {text: $hypothesis})
                        MERGE (h)-[:RELATED_TO]->(p)
                        """,
                        id=paper['id'], source=paper['source'], title=paper['title'], hypothesis=hypothesis
                    )
            logger.info(f"Updated Neo4j graph for hypothesis: {hypothesis}")
        except Exception as e:
            logger.error(f"Error updating Neo4j graph for hypothesis {hypothesis}: {str(e)}")

    def compute_novelty(self, hypothesis: str) -> float:
        """Compute novelty score based on related papers in Neo4j graph."""
        try:
            if not self.driver:
                logger.error("No Neo4j connection available")
                return 0.5
            
            with self.driver.session(database=self.db_name) as session:
                result = session.run(
                    """
                    MATCH (h:Hypothesis {text: $hypothesis})-[:RELATED_TO]->(p:Paper)
                    RETURN count(p) as paper_count
                    """,
                    hypothesis=hypothesis
                )
                paper_count = result.single()['paper_count'] if result.single() else 0
                
                # Novelty score: inversely proportional to number of related papers
                # Fewer papers = higher novelty
                novelty = 1.0 / (1.0 + paper_count) if paper_count > 0 else 1.0
                return max(0.0, min(1.0, novelty))
        except Exception as e:
            logger.error(f"Error computing novelty for hypothesis {hypothesis}: {str(e)}")
            return 0.5  # Default novelty score

    def cross_reference_hypotheses(self) -> List[Dict]:
        """Cross-reference hypotheses and compute novelty scores."""
        try:
            hypotheses = self.load_hypotheses()
            if not hypotheses:
                logger.warning("No hypotheses to cross-reference")
                return []
            
            results = []
            for hyp in hypotheses:
                hypothesis_text = hyp.get('hypothesis', '')
                if not hypothesis_text:
                    logger.warning("Skipping hypothesis with no text")
                    continue
                
                # Query APIs
                pubmed_papers = self.query_pubmed(hypothesis_text)
                arxiv_papers = self.query_arxiv(hypothesis_text)
                all_papers = pubmed_papers + arxiv_papers
                
                # Update Neo4j graph
                self.update_graph(hypothesis_text, all_papers)
                
                # Compute novelty
                novelty = self.compute_novelty(hypothesis_text)
                
                # Update hypothesis with novelty score
                hyp['novelty_api'] = float(novelty)
                results.append(hyp)
                
                logger.info(f"Hypothesis: {hypothesis_text}, Novelty: {novelty:.4f}")
            
            # Save results
            self.save_results(results)
            return results
        except Exception as e:
            logger.error(f"Error cross-referencing hypotheses: {str(e)}")
            return []

    def save_results(self, results: List[Dict]) -> None:
        """Save cross-referenced hypotheses with novelty scores to JSON."""
        try:
            output_file = self.hypothesis_file.parent / 'cross_referenced_hypotheses.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved cross-referenced hypotheses to {output_file}")
        except Exception as e:
            logger.error(f"Error saving cross-referenced hypotheses: {str(e)}")

    def run(self) -> Optional[List[Dict]]:
        """Run cross-referencing process for all hypotheses."""
        try:
            return self.cross_reference_hypotheses()
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage for testing
    api = ResearchAPI(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )
    results = api.run()
    if results:
        logger.info(f"Cross-referenced hypotheses: {json.dumps(results, indent=2)}")
    api.close()
