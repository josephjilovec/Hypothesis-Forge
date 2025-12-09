#!/bin/bash
# Production entrypoint script for Hypothesis Forge

set -e

echo "Starting Hypothesis Forge..."

# Wait for Neo4j if configured
if [ -n "$NEO4J_URI" ] && [ "$NEO4J_URI" != "bolt://localhost:7687" ]; then
    echo "Waiting for Neo4j to be ready..."
    # Simple wait - in production, use a proper health check
    sleep 5
fi

# Run health check API in background if enabled
if [ "$ENABLE_HEALTH_API" = "true" ]; then
    echo "Starting health check API..."
    python -m api.health &
    HEALTH_API_PID=$!
fi

# Run Streamlit app
echo "Starting Streamlit dashboard..."
exec streamlit run frontend/app.py \
    --server.port=${STREAMLIT_SERVER_PORT:-8501} \
    --server.address=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0} \
    --server.enableStaticServing \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true

