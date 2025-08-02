Hypothesis Forge API Endpoints Documentation
This document details the PubMed and arXiv API endpoints used in the knowledge/research_api.py script of the Hypothesis Forge project. These APIs are queried to cross-reference hypotheses for novelty by fetching relevant research papers. Below, we describe each endpoint, including URLs, parameters, response formats, and example queries, ensuring clarity for developers integrating or extending the system.
1. PubMed API
The PubMed API, provided by NCBI's Entrez E-Utilities, is used to fetch biomedical research papers. It consists of two main endpoints: esearch for searching papers and esummary for retrieving paper details.
1.1. ESearch Endpoint
Purpose: Search PubMed for papers matching a query term.

URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
Method: GET
Parameters:
db: Database to search (set to pubmed).
term: Search query (e.g., keywords from a hypothesis).
retmax: Maximum number of results to return (e.g., 5).
retmode: Response format (set to json).


Response Format: JSON
esearchresult.idlist: List of PubMed IDs (PMIDs) for matching papers.


Example Query:https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=protein+stability&retmax=5&retmode=json


Example Response:{
  "header": {
    "type": "esearch",
    "version": "0.3"
  },
  "esearchresult": {
    "count": "12345",
    "retmax": "5",
    "retstart": "0",
    "idlist": ["12345678", "23456789", "34567890", "45678901", "56789012"],
    ...
  }
}



1.2. ESummary Endpoint
Purpose: Retrieve summary details for specific PubMed papers.

URL: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi
Method: GET
Parameters:
db: Database (set to pubmed).
id: Comma-separated list of PMIDs.
retmode: Response format (set to json).


Response Format: JSON
result.<PMID>: Details for each paper, including title and keywords (if available).


Example Query:https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=12345678&retmode=json


Example Response:{
  "header": {
    "type": "esummary",
    "version": "0.3"
  },
  "result": {
    "12345678": {
      "uid": "12345678",
      "title": "Impact of Temperature on Protein Stability",
      "keywords": ["protein", "stability", "temperature"],
      ...
    },
    ...
  }
}



Usage in research_api.py

The query_pubmed method:
Uses esearch to find PMIDs for a hypothesis (e.g., "protein stability").
Uses esummary to fetch details for each PMID.
Extracts id, title, and keywords (as topics) for Neo4j graph updates.


Error handling: Catches HTTP errors, timeouts, and JSON parsing issues, returning an empty list on failure.

2. arXiv API
The arXiv API is used to fetch physics and astrophysics papers for hypothesis cross-referencing.
2.1. Query Endpoint
Purpose: Search arXiv for papers matching a query term.

URL: http://export.arxiv.org/api/query
Method: GET
Parameters:
search_query: Search term (e.g., hypothesis keywords).
max_results: Maximum number of results to return (e.g., 5).


Response Format: XML (Atom format)
<entry> elements contain paper details: <id>, <title>, and <category> (topics).


Example Query:http://export.arxiv.org/api/query?search_query=gravitational+field&max_results=5


Example Response:<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678</id>
    <title>Gravitational Field Effects on Orbital Dynamics</title>
    <category term="astro-ph.CO"/>
  </entry>
  ...
</feed>



Usage in research_api.py

The query_arxiv method:
Queries the arXiv API with a hypothesis (e.g., "gravitational field").
Parses the XML response using xml.etree.ElementTree.
Extracts id (arXiv ID), title, and category (as topics) for Neo4j graph updates.


Error handling: Handles HTTP errors, timeouts, and XML parsing issues, returning an empty list on failure.

Notes for Developers

Authentication: Neither API requires authentication, but rate limits apply (PubMed: 3 requests/second without API key; arXiv: no strict limit but be cautious).
Error Handling: Both query_pubmed and query_arxiv methods in research_api.py include robust error handling, logging issues and returning empty lists to ensure system stability.
Data Storage: Results are stored in a Neo4j knowledge graph via graph_builder.py, with hypotheses linked to papers and topics.
Extending APIs:
To add new APIs (e.g., Google Scholar), implement a similar query method in research_api.py and update update_graph to handle new data formats.
Ensure new APIs support JSON or XML responses for compatibility with existing parsing logic.


Testing: Use tests/test_api.py to mock API responses with unittest.mock and test integration with Neo4j. Avoid live API calls in tests to ensure reproducibility.

Example Query Workflow

Hypothesis: "Increase temperature affects protein stability"
PubMed Query:
ESearch: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=protein+stability&retmax=5&retmode=json
ESummary (for PMID 12345678): https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id=12345678&retmode=json
Result: List of papers with titles and keywords.


arXiv Query:
Query: http://export.arxiv.org/api/query?search_query=protein+stability&max_results=5
Result: List of papers with titles and categories.


Neo4j Update: Store papers as nodes and link to the hypothesis node.
Novelty Score: Computed based on the number of related papers in the Neo4j graph (inversely proportional).

This documentation provides a clear reference for developers to understand and extend the API interactions in Hypothesis Forge, ensuring seamless integration with the knowledge graph and hypothesis ranking system.
