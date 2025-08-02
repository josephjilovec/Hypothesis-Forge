# Use Python 3.9 slim base image for minimal size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Streamlit (default 8501)
EXPOSE 8501

# Set environment variables for Streamlit
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run Streamlit with static file serving
CMD ["streamlit", "run", "frontend/app.py", "--server.enableStaticServing", "--server.port", "8501"]
