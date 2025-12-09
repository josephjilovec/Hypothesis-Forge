"""
Research API integration for cross-referencing hypotheses.
Integrates with PubMed and arXiv APIs to check novelty.
"""
import logging
import json
import time
from typing import List, Dict, Any, Optional
import requests
import feedparser

from config.config import PUBMED_API_KEY, ARXIV_API_BASE, HYPOTHESIS_DIR
from utils.logging_config import logger
from utils.rate_limiter import pubmed_limiter, arxiv_limiter


class PubMedClient:
    """Client for PubMed API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PubMed client.

        Args:
            api_key: Optional PubMed API key
        """
        self.api_key = api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search PubMed for articles.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of article dictionaries
        """
        # Rate limiting
        pubmed_limiter.wait_if_needed("pubmed_search")
        try:
            # Search endpoint
            search_url = f"{self.base_url}/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return []

            # Fetch article details
            fetch_url = f"{self.base_url}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids[:5]),  # Limit to 5 for performance
                "retmode": "xml",
            }
            if self.api_key:
                fetch_params["api_key"] = self.api_key

            # For simplicity, return basic info
            articles = []
            for pmid in pmids[:max_results]:
                articles.append({
                    "pmid": pmid,
                    "title": f"PubMed article {pmid}",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                })

            return articles

        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
            return []

    def check_novelty(self, hypothesis_text: str) -> Dict[str, Any]:
        """
        Check hypothesis novelty against PubMed.

        Args:
            hypothesis_text: Hypothesis text to check

        Returns:
            Novelty assessment dictionary
        """
        # Extract key terms from hypothesis
        keywords = self._extract_keywords(hypothesis_text)
        query = " AND ".join(keywords[:3])  # Use top 3 keywords

        articles = self.search(query, max_results=5)

        return {
            "similar_articles_count": len(articles),
            "novelty_score": max(0.0, 1.0 - (len(articles) * 0.2)),
            "similar_articles": articles[:3],  # Top 3
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:10]


class ArxivClient:
    """Client for arXiv API."""

    def __init__(self):
        """Initialize arXiv client."""
        self.base_url = ARXIV_API_BASE

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of paper dictionaries
        """
        # Rate limiting
        arxiv_limiter.wait_if_needed("arxiv_search")
        try:
            params = {
                "search_query": query,
                "start": 0,
                "max_results": max_results,
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            feed = feedparser.parse(response.content)

            papers = []
            for entry in feed.entries[:max_results]:
                papers.append({
                    "id": entry.id.split("/")[-1],
                    "title": entry.title,
                    "authors": [author.name for author in entry.authors],
                    "published": entry.published,
                    "summary": entry.summary[:200] + "..." if len(entry.summary) > 200 else entry.summary,
                    "url": entry.link,
                })

            return papers

        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")
            return []

    def check_novelty(self, hypothesis_text: str) -> Dict[str, Any]:
        """
        Check hypothesis novelty against arXiv.

        Args:
            hypothesis_text: Hypothesis text to check

        Returns:
            Novelty assessment dictionary
        """
        keywords = self._extract_keywords(hypothesis_text)
        query = " OR ".join(keywords[:3])

        papers = self.search(query, max_results=5)

        return {
            "similar_papers_count": len(papers),
            "novelty_score": max(0.0, 1.0 - (len(papers) * 0.15)),
            "similar_papers": papers[:3],
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:10]


class ResearchAPIClient:
    """Combined research API client."""

    def __init__(self):
        """Initialize research API client."""
        self.pubmed = PubMedClient(api_key=PUBMED_API_KEY)
        self.arxiv = ArxivClient()

    def cross_reference_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cross-reference hypothesis with PubMed and arXiv.

        Args:
            hypothesis: Hypothesis dictionary

        Returns:
            Cross-referenced hypothesis with novelty scores
        """
        hypothesis_text = hypothesis.get("text", "")

        logger.info(f"Cross-referencing hypothesis: {hypothesis_text[:50]}...")

        # Check PubMed
        pubmed_result = self.pubmed.check_novelty(hypothesis_text)

        # Check arXiv
        arxiv_result = self.arxiv.check_novelty(hypothesis_text)

        # Combine results
        combined_novelty = (pubmed_result["novelty_score"] + arxiv_result["novelty_score"]) / 2.0

        hypothesis["cross_reference"] = {
            "pubmed": pubmed_result,
            "arxiv": arxiv_result,
            "combined_novelty_score": combined_novelty,
        }

        # Update overall novelty score
        original_novelty = hypothesis.get("novelty_score", 0.5)
        hypothesis["novelty_score"] = (original_novelty + combined_novelty) / 2.0

        return hypothesis


def cross_reference_hypotheses(hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cross-reference multiple hypotheses.

    Args:
        hypotheses: List of hypothesis dictionaries

    Returns:
        List of cross-referenced hypotheses
    """
    logger.info(f"Cross-referencing {len(hypotheses)} hypotheses...")

    client = ResearchAPIClient()
    cross_referenced = []

    for i, hypothesis in enumerate(hypotheses):
        try:
            cross_ref_hyp = client.cross_reference_hypothesis(hypothesis)
            cross_referenced.append(cross_ref_hyp)
            logger.info(f"Cross-referenced hypothesis {i+1}/{len(hypotheses)}")
        except Exception as e:
            logger.error(f"Failed to cross-reference hypothesis {i+1}: {e}")
            cross_referenced.append(hypothesis)  # Include original if cross-ref fails

    # Save results
    output_path = HYPOTHESIS_DIR / "cross_referenced_hypotheses.json"
    with open(output_path, "w") as f:
        json.dump(cross_referenced, f, indent=2)

    logger.info(f"Saved cross-referenced hypotheses to {output_path}")
    return cross_referenced


if __name__ == "__main__":
    # Example usage
    sample_hypotheses = [
        {
            "id": "hyp_1",
            "text": "Structural hypothesis: The protein structure exhibits characteristics that may influence binding.",
            "novelty_score": 0.8,
        }
    ]

    cross_referenced = cross_reference_hypotheses(sample_hypotheses)
    logger.info(f"Cross-referenced {len(cross_referenced)} hypotheses")

