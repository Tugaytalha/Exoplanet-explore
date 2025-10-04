# Dockerfile for Exoplanet Explorer API
# Fetches data, trains model, and runs FastAPI backend

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir uvicorn[standard]

# Copy application files
COPY fetch.py .
COPY api.py .
COPY train_koi_disposition.py .

# Create data and model directories
RUN mkdir -p data model_outputs

# Expose port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "================================"\n\
echo "Exoplanet Explorer API - Startup"\n\
echo "================================"\n\
\n\
# Check if data exists\n\
if [ ! -f "data/koi_with_relative_location.csv" ]; then\n\
    echo ""\n\
    echo "Step 1: Fetching data from NASA..."\n\
    python fetch.py\n\
    echo "✅ Data fetched successfully!"\n\
else\n\
    echo ""\n\
    echo "✅ Data already exists, skipping fetch"\n\
fi\n\
\n\
# Check if model exists\n\
if [ ! -f model_outputs/xgboost_koi_disposition_*.joblib ] 2>/dev/null; then\n\
    if [ "${SKIP_TRAINING}" = "true" ]; then\n\
        echo ""\n\
        echo "⚠️  Skipping model training (SKIP_TRAINING=true)"\n\
        echo "   API will use dataset dispositions as fallback"\n\
    else\n\
        echo ""\n\
        echo "Step 2: Training XGBoost model..."\n\
        echo "This will take 5-15 minutes..."\n\
        python train_koi_disposition.py\n\
        echo "✅ Model trained successfully!"\n\
    fi\n\
else\n\
    echo ""\n\
    echo "✅ Model already exists, skipping training"\n\
fi\n\
\n\
echo ""\n\
echo "Step 3: Starting API server..."\n\
echo "API will be available at http://localhost:8000"\n\
echo "Documentation at http://localhost:8000/docs"\n\
echo ""\n\
\n\
# Start API\n\
exec uvicorn api:app --host 0.0.0.0 --port 8000\n\
' > /app/startup.sh && chmod +x /app/startup.sh

# Set startup script as entrypoint
ENTRYPOINT ["/app/startup.sh"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Labels
LABEL maintainer="Exoplanet Explorer Team" \
      description="NASA Exoplanet Classification API with XGBoost ML" \
      version="3.0.0"

