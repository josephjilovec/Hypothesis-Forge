# Quick Start Guide

Get Hypothesis Forge up and running in minutes!

## Prerequisites

- Python 3.9+
- pip
- (Optional) Docker and Docker Compose

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/josephjilovec/Hypothesis-Forge.git
   cd Hypothesis-Forge
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment (optional):**
   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

## Running the Pipeline

### Option 1: Run Complete Pipeline

```bash
python run_pipeline.py
```

This will:
- Preprocess data (creates sample data if needed)
- Run simulations
- Generate hypotheses using RL
- Rank hypotheses
- Cross-reference with research APIs

### Option 2: Run Individual Components

1. **Preprocess data:**
   ```bash
   python data/preprocess.py
   ```

2. **Run simulations:**
   ```bash
   python simulation/sim_engine.py
   ```

3. **Generate hypotheses:**
   ```bash
   python hypothesis/agent.py
   ```

4. **Rank hypotheses:**
   ```bash
   python hypothesis/hypothesis_ranking.py
   ```

5. **Cross-reference:**
   ```bash
   python knowledge/research_api.py
   ```

## Launch Dashboard

```bash
streamlit run frontend/app.py
```

Open your browser to `http://localhost:8501`

## Docker Quick Start

```bash
docker-compose -f deploy/docker-compose.yml up
```

Access dashboard at `http://localhost:8501`

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [PRODUCTION.md](PRODUCTION.md) for deployment guides
- Review [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Troubleshooting

### Neo4j Connection Issues
- The system works without Neo4j (offline mode)
- To enable knowledge graph features, install Neo4j and configure in `.env`

### Import Errors
- Ensure you're in the project root directory
- Activate your virtual environment
- Install all dependencies: `pip install -r requirements.txt`

### Dashboard Not Loading
- Check that Streamlit is installed: `pip install streamlit`
- Verify port 8501 is not in use
- Check logs for error messages

## Getting Help

- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the [AUDIT_REPORT.md](AUDIT_REPORT.md) for architecture details

