Hypothesis Forge

Hypothesis Forge is an AI-driven system designed to accelerate scientific discovery by generating and ranking novel hypotheses from complex datasets, such as AlphaFold protein structures and NASA astrophysical data. Aligned with xAI's mission to advance human scientific understanding, it integrates data ingestion, simulation, reinforcement learning (RL), knowledge graph management, and an interactive dashboard to provide a robust pipeline for scientific hypothesis generation. The system is modular, containerized with Docker, and deployable on cloud platforms like AWS or DigitalOcean, ensuring scalability and reliability.

Project Overview

Hypothesis Forge processes raw scientific datasets, runs simulations (e.g., protein folding, gravitational dynamics), generates hypotheses using a Proximal Policy Optimization (PPO) RL agent, cross-references them for novelty via PubMed and arXiv APIs, and visualizes results in a Streamlit-based dashboard. Key features include:

Data Ingestion: Preprocesses raw datasets (PDB, FITS) using Pandas, BioPython, and Astropy.
Simulation Engine: Executes AI-driven (PyTorch) or physics-based simulations, optimized for <5-minute runtime.
RL Hypothesis Generation: Uses a PPO agent to propose novel "what-if" scenarios based on simulation outputs.
Knowledge Graph: Builds a Neo4j graph to store research relationships and compute novelty scores.
Interactive Dashboard: Displays simulation results and ranked hypotheses with Plotly visualizations and JavaScript interactivity.
Deployment: Containerized with Docker and deployable on AWS Elastic Beanstalk with auto-scaling.

The repository is structured for modularity, with comprehensive tests (>80% coverage) and CI/CD via GitHub Actions.

Setup Instructions
Prerequisites

Python: 3.9
Docker: Latest version for containerization (optional)
Neo4j: Community edition for knowledge graph
Git: For cloning the repository

Installation

Clone the Repository:
git clone https://github.com/josephjilovec/Hypothesis-Forge.git
cd Hypothesis-Forge


Install Dependencies:
pip install -r requirements.txt

Dependencies include numpy, pandas, torch, biopython, astropy, neo4j, requests, streamlit, plotly, and pytest.

Set Up Environment:

Copy env.example to .env and configure your settings:
cp env.example .env

Edit .env and set:
- Neo4j credentials (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
- API keys (PUBMED_API_KEY)
- Other configuration options

Set Up Neo4j:

Install Neo4j Community Edition locally or use a cloud-hosted instance (e.g., Neo4j Aura).
The system will work without Neo4j but knowledge graph features will be disabled.




Quick Start:

Run the complete pipeline:
python run_pipeline.py

This will:
1. Preprocess data (creates sample data if none exists)
2. Run simulations
3. Generate hypotheses
4. Rank hypotheses
5. Cross-reference with research APIs
6. Build knowledge graph (if Neo4j is available)

Then launch the dashboard:
streamlit run frontend/app.py

Prepare Data:

Place raw datasets in data/raw/alphafold/ (PDB files) and data/raw/nasa/ (FITS files).
Run data/preprocess.py to generate processed data:
python data/preprocess.py





Running Locally
You can run Hypothesis Forge locally using either Python directly or Docker.
Option 1: Run with Python

Start the Streamlit dashboard:streamlit run frontend/app.py --server.enableStaticServing


Access the dashboard at http://localhost:8501.
Navigate to "Simulations" to view results or "Hypotheses" to see ranked hypotheses.

Option 2: Run with Docker

Build the Docker image:docker build -t hypothesis-forge -f deploy/Dockerfile .


Run the container:docker run -p 8501:8501 hypothesis-forge


Access the dashboard at http://localhost:8501.

Deploying to the Cloud
To deploy Hypothesis Forge to a cloud platform for public access, use AWS Elastic Beanstalk or DigitalOcean. Note that cloud deployment incurs costs (e.g., ~$12-$50/month for AWS or DigitalOcean, plus Neo4j hosting). Follow these steps:
AWS Elastic Beanstalk

Install AWS CLI: pip install awscli.
Configure AWS credentials:aws configure

Enter your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and set region to us-east-1.
Build and push the Docker image to Docker Hub:docker build -t hypothesis-forge -f deploy/Dockerfile .
docker tag hypothesis-forge:latest your_docker_username/hypothesis-forge:latest
docker push your_docker_username/hypothesis-forge:latest


Initialize and deploy to Elastic Beanstalk:eb init -p docker hypothesis-forge --region us-east-1
eb create hypothesis-forge-env --single --instance_type t3.medium
eb deploy hypothesis-forge-env


Set environment variables for Streamlit and Neo4j:eb setenv STREAMLIT_SERVER_PORT=8501 STREAMLIT_SERVER_ADDRESS=0.0.0.0 PYTHONUNBUFFERED=1

Configure Neo4j credentials in the AWS Elastic Beanstalk console.
Access the deployed app at http://hypothesis-forge-env.<random-id>.us-east-1.elasticbeanstalk.com.

DigitalOcean

Create a DigitalOcean Droplet or use App Platform (~$12-$20/month).
Install Docker on the Droplet:sudo apt update
sudo apt install docker.io
sudo systemctl start docker


Build and run the Docker image:docker build -t hypothesis-forge -f deploy/Dockerfile .
docker run -p 8501:8501 hypothesis-forge


Access at http://<droplet-ip>:8501.

Usage Examples
1. Preprocessing Data
Process raw datasets:
python data/preprocess.py

Outputs cleaned data to data/processed/ (e.g., test_processed.parquet).
2. Running Simulations
Execute simulations:
python simulation/sim_engine.py

Saves results to simulation/results/ (e.g., test_processed_results.parquet).
3. Generating and Ranking Hypotheses
Generate and rank hypotheses:
python hypothesis/agent.py  # Requires environment setup
python hypothesis/hypothesis_ranking.py

Saves ranked hypotheses to hypothesis/ranked_hypotheses.json.
4. Cross-Referencing with APIs
Check hypothesis novelty:
python knowledge/research_api.py

Saves results to hypothesis/cross_referenced_hypotheses.json.
5. Viewing the Dashboard
Launch the dashboard:
streamlit run frontend/app.py --server.enableStaticServing

View simulations and hypotheses at http://localhost:8501.
Testing
Run unit and integration tests (>80% coverage):
pytest tests/ -v

Contributing
Contributions are welcome under the MIT License. Please:

Fork the repository and create a feature branch (feature/your-feature).
Submit pull requests to the develop branch.
Ensure tests pass and add new tests for new features.
Follow code style (flake8, max line length 88).

License
Hypothesis Forge is licensed under the MIT License. See LICENSE for details.
Contact
For questions or issues, contact Joseph Jilovec or open an issue on GitHub.
