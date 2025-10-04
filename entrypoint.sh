#!/bin/bash
set -e

echo "🚀 Starting Exoplanet Classification API..."

# Check if data file exists
if [ ! -f "data/koi_with_relative_location.csv" ]; then
    echo "📥 Data file not found. Running fetch.py to download data..."
    python fetch.py
    echo "✅ Data download complete!"
else
    echo "✅ Data file already exists, skipping fetch.py"
fi

# Check if model files exist
MODEL_COUNT=$(find model_outputs -name "xgboost_koi_disposition_*.joblib" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "⚠️  No trained model found in model_outputs/"
    echo "   The API will still run but predictions will use dataset dispositions."
    echo "   To train a model, run: docker-compose exec exoplanet-api python train_koi_disposition.py"
else
    echo "✅ Model files found: $MODEL_COUNT model(s)"
fi

echo ""
echo "🌟 Starting FastAPI server on http://0.0.0.0:8000"
echo "   Access API documentation at http://localhost:8000/docs"
echo ""

# Start the FastAPI application
exec uvicorn api:app --host 0.0.0.0 --port 8000
