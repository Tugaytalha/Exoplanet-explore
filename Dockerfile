# Dockerfile for Exoplanet Classification API
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api.py .
COPY fetch.py .
COPY predict_koi_disposition.py .
COPY train_koi_disposition.py .
COPY example_usage.py .
COPY test_model_loading.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create directories for data and models
# (will be populated by volumes or fetch.py at runtime)
RUN mkdir -p data model_outputs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Use entrypoint script to run fetch.py before starting API
ENTRYPOINT ["./entrypoint.sh"]
