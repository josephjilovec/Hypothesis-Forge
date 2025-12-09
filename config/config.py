"""
Configuration management for Hypothesis Forge.
Handles environment variables and application settings.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIMULATION_DIR = BASE_DIR / "simulation"
RESULTS_DIR = SIMULATION_DIR / "results"
HYPOTHESIS_DIR = BASE_DIR / "hypothesis"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# Neo4j configuration
NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")

# API configuration
PUBMED_API_KEY: Optional[str] = os.getenv("PUBMED_API_KEY")
ARXIV_API_BASE: str = "http://export.arxiv.org/api/query"

# Simulation configuration
SIMULATION_TIMEOUT: int = int(os.getenv("SIMULATION_TIMEOUT", "300"))  # 5 minutes
MAX_SIMULATION_RUNTIME: int = int(os.getenv("MAX_SIMULATION_RUNTIME", "300"))

# RL Agent configuration
RL_LEARNING_RATE: float = float(os.getenv("RL_LEARNING_RATE", "3e-4"))
RL_BATCH_SIZE: int = int(os.getenv("RL_BATCH_SIZE", "64"))
RL_GAMMA: float = float(os.getenv("RL_GAMMA", "0.99"))
RL_EPISODES: int = int(os.getenv("RL_EPISODES", "1000"))

# Streamlit configuration
STREAMLIT_SERVER_PORT: int = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS: str = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")

# Logging configuration
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

# Model paths
MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Create necessary directories
for directory in [
    RAW_DATA_DIR / "alphafold",
    RAW_DATA_DIR / "nasa",
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    HYPOTHESIS_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

