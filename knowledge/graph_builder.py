import logging
from neo4j import GraphDatabase
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jGraphBuilder:
    """Builds a Neo4j knowledge graph for research relationships using PubMed and arXiv data."""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password", db_name: str = "neo4j"):
        self.uri = neo4j_uri
        self.user = neo4j_user
        self.password = neo4j_password
        self.db_name = db_name
        self.driver = None
        self.connect()
        logger.info("Initialized Neo4jGraphBuilder")

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

    def fetch_pubmed_data(self, query: str, max_results: int = 5) -> List[Dict]:
        """Fetch research papers from PubMed API."""
        try:
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
                    'topics': doc.get('keywords', []) or ['unknown'],
                    'citations': []  # PubMed API may not provide citations directly
                })
            return papers
        except Exception as e:
            logger.error(f"Error fetching PubMed data for query {query}: {str(e)}")
            return []

    def fetch_arxiv_data(self, query: str, max_results: int = 5) -> List[Dict]:
        """Fetch research papers from arXiv API."""
        try:
            url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                title = entry.find('{http://www.w3.org/2005/Atom}title').text
                categories = [cat.get('term') for cat in entry.findall('{http://www.w3.org/2005/Atom}category')]
                
                papers.append({
                    'id': arxiv_id,
                    'title': title,
                    'topics': categories or ['unknown'],
                    'citations': []  # arXiv API does not provide citation data
                })
            return papers
        except Exception as e:
            logger.error(f"Error fetching arXiv data for query {query}: {str(e)}")
            return []

    def create_nodes(self, session, papers: List[Dict], source: str) -> None:
        """Create Paper and Topic nodes in Neo4j."""
        try:
            for paper in papers:
                # Create Paper node
                session.run(
                    """
                    MERGE (p:Paper {id: $id, source: $source})
                    SET p.title = $title
                    """,
                    id=paper['id'], source=source, title=paper['title']
                )
                
                # Create Topic nodes and relationships
                for topic in paper['topics']:
                    session.run(
                        """
                        MERGE (t:Topic {name: $topic})
                        MERGE (p:Paper {id: $id, source: $source})
                        MERGE (p)-[:RELATED_TO]->(t)
                        """,
                        topic=topic, id=paper['id'], source=source
                    )
            logger.info(f"Created nodes for {len(papers)} papers from {source}")
        except Exception as e:
            logger.error(f"Error creating nodes for {source}: {str(e)}")

    def create_citation_relationships(self, session, papers: List[Dict], source: str) -> None:
        """Create citation relationships between papers."""
        try:
            for paper in papers:
                for citation_id in paper['citations']:
                    session.run(
                        """
                        MERGE (p1:Paper {id: $id, source: $source})
                        MERGE (p2:Paper {id: $citation_id})
                        MERGE (p1)-[:CITES]->(p2)
                        """,
                        id=paper['id'], source=source, citation_id=citation_id
                    )
            logger.info(f"Created citation relationships for {source}")
        except Exception as e:
            logger.error(f"Error creating citation relationships for {source}: {str(e)}")

    def build_graph(self, query: str, max_results: int = 5) -> Optional[Dict]:
        """Build knowledge graph from PubMed and arXiv data."""
        try:
            if not self.driver:
                logger.error("No Neo4j connection available")
                return None

            # Fetch data
            pubmed_papers = self.fetch_pubmed_data(query, max_results)
            arxiv_papers = self.fetch_arxiv_data(query, max_results)
            
            # Combine papers
            all_papers = pubmed_papers + arxiv_papers
            
            if not all_papers:
                logger.warning("No papers fetched for query {}".format(query))
                return None

            # Create nodes and relationships in Neo4j
            with self.driver.session(database=self.db_name) as session:
                self.create_nodes(session, pubmed_papers, 'PubMed')
                self.create_nodes(session, arxiv_papers, 'arXiv')
                self.create_citation_relationships(session, pubmed_papers, 'PubMed')
                self.create_citation_relationships(session, arxiv_papers, 'arXiv')

            # Return summary
            summary = {
                'query': query,
                'num_papers': len(all_papers),
                'pubmed_papers': len(pubmed_papers),
                'arxiv_papers': len(arxiv_papers)
            }
            logger.info(f"Built graph with {summary['num_papers']} papers")
            return summary

        except Exception as e:
            logger.error(f"Error building graph: {str(e)}")
            return None

    def run(self, query: str, max_results: int = 5) -> Optional[Dict]:
        """Run graph building process for a given query."""
        try:
            return self.build_graph(query, max_results)
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage for testing
    graph_builder = Neo4jGraphBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )
    result = graph_builder.run(query="protein folding", max_results=3)
    if result:
        logger.info(f"Graph building result: {result}")
    graph_builder.close()
