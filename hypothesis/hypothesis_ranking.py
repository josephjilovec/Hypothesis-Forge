import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HypothesisRanker:
    """Ranks hypotheses based on novelty and feasibility metrics."""
    
    def __init__(self, history_file: str = 'hypothesis/history.json', output_file: str = 'hypothesis/ranked_hypotheses.json'):
        self.history_file = Path(history_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.hypothesis_history: List[Dict] = self.load_history()
        self.embedding_cache = {}  # Cache for hypothesis embeddings
        logger.info("Initialized HypothesisRanker")

    def load_history(self) -> List[Dict]:
        """Load hypothesis history from JSON file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading hypothesis history: {str(e)}")
            return []

    def save_history(self, hypotheses: List[Dict]) -> None:
        """Save hypothesis history to JSON file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(hypotheses, f, indent=2)
            logger.info(f"Saved hypothesis history to {self.history_file}")
        except Exception as e:
            logger.error(f"Error saving hypothesis history: {str(e)}")

    def compute_novelty(self, hypothesis: str, state: np.ndarray) -> float:
        """Compute novelty score using cosine similarity of state embeddings."""
        try:
            # Simple embedding: use state vector as embedding (can be replaced with NLP embeddings)
            new_embedding = state / (np.linalg.norm(state) + 1e-6)  # Normalize
            
            if not self.hypothesis_history:
                return 1.0  # First hypothesis is maximally novel
            
            # Compute similarity with historical embeddings
            similarities = []
            for hist in self.hypothesis_history:
                hist_embedding = self.embedding_cache.get(hist['hypothesis'])
                if hist_embedding is None:
                    hist_embedding = np.array(hist['state']) / (np.linalg.norm(hist['state']) + 1e-6)
                    self.embedding_cache[hist['hypothesis']] = hist_embedding
                similarity = cosine_similarity([new_embedding], [hist_embedding])[0][0]
                similarities.append(similarity)
            
            # Novelty is inverse of max similarity
            novelty = 1.0 - max(similarities) if similarities else 1.0
            return max(0.0, min(1.0, novelty))
        except Exception as e:
            logger.error(f"Error computing novelty for {hypothesis}: {str(e)}")
            return 0.5  # Default novelty score

    def compute_feasibility(self, hypothesis: str, state: np.ndarray) -> float:
        """Compute feasibility score based on predefined scientific criteria."""
        try:
            # Example criteria: state magnitude and hypothesis-specific rules
            state_norm = np.linalg.norm(state)
            max_norm = 10.0  # Example threshold for reasonable state magnitude
            base_feasibility = min(1.0, state_norm / max_norm)
            
            # Domain-specific feasibility adjustments
            if "protein stability" in hypothesis.lower():
                # Example: higher feasibility if state norm is low (stable structure)
                feasibility = base_feasibility * 0.8 if state_norm < 5.0 else base_feasibility * 0.5
            elif "gravitational field" in hypothesis.lower():
                # Example: higher feasibility if state norm is high (dynamic system)
                feasibility = base_feasibility * 1.2 if state_norm > 5.0 else base_feasibility * 0.7
            else:
                feasibility = base_feasibility
            
            return max(0.0, min(1.0, feasibility))
        except Exception as e:
            logger.error(f"Error computing feasibility for {hypothesis}: {str(e)}")
            return 0.5  # Default feasibility score

    def rank_hypotheses(self, hypotheses: List[Dict]) -> List[Dict]:
        """Rank hypotheses based on combined novelty and feasibility scores."""
        try:
            ranked_hypotheses = []
            for hyp in hypotheses:
                hypothesis_text = hyp.get('hypothesis', '')
                state = np.array(hyp.get('state', []), dtype=np.float32)
                
                if not hypothesis_text or len(state) == 0:
                    logger.warning(f"Skipping invalid hypothesis: {hypothesis_text}")
                    continue
                
                novelty = self.compute_novelty(hypothesis_text, state)
                feasibility = self.compute_feasibility(hypothesis_text, state)
                score = 0.7 * novelty + 0.3 * feasibility  # Weighted score
                
                ranked_hypotheses.append({
                    'hypothesis': hypothesis_text,
                    'state': state.tolist(),
                    'novelty': float(novelty),
                    'feasibility': float(feasibility),
                    'score': float(score)
                })
                
                # Update history
                self.hypothesis_history.append({
                    'hypothesis': hypothesis_text,
                    'state': state.tolist()
                })
            
            # Sort by score in descending order
            ranked_hypotheses.sort(key=lambda x: x['score'], reverse=True)
            
            # Save ranked hypotheses
            self.save_ranked_hypotheses(ranked_hypotheses)
            self.save_history(self.hypothesis_history)
            
            return ranked_hypotheses
        except Exception as e:
            logger.error(f"Error ranking hypotheses: {str(e)}")
            return []

    def save_ranked_hypotheses(self, ranked_hypotheses: List[Dict]) -> None:
        """Save ranked hypotheses to JSON file for dashboard."""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(ranked_hypotheses, f, indent=2)
            logger.info(f"Saved ranked hypotheses to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving ranked hypotheses: {str(e)}")

    def run(self, hypotheses: List[Dict]) -> Optional[List[Dict]]:
        """Process and rank a list of hypotheses."""
        try:
            if not hypotheses:
                logger.warning("No hypotheses provided for ranking")
                return None
            return self.rank_hypotheses(hypotheses)
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage for testing
    ranker = HypothesisRanker()
    test_hypotheses = [
        {
            'hypothesis': "Increase temperature affects protein stability",
            'state': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        },
        {
            'hypothesis': "Alter gravitational field changes orbital dynamics",
            'state': [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        }
    ]
    ranked = ranker.run(test_hypotheses)
    if ranked:
        logger.info("Ranked hypotheses: {}".format(json.dumps(ranked, indent=2)))
