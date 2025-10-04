# 🌟 Kepler KOI Exoplanet Classification API

## Overview

This FastAPI application provides access to NASA's Kepler Objects of Interest (KOI) dataset with **real disposition classifications** and optional **ML model predictions** using the trained XGBoost classifier.

### What's New in Version 2.0
- ✅ **Real Exoplanet Disposition Data** (no more dummy flags!)
- ✅ **XGBoost ML Model Integration** for predictions
- ✅ **Three Classification Categories**: CONFIRMED, FALSE_POSITIVE, CANDIDATE
- ✅ **Confidence Scores** for each prediction
- ✅ **Advanced Filtering** by disposition type
- ✅ **Dataset Statistics** endpoint
- ✅ **Model Status** monitoring

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy joblib
```

### 2. Run the API

```bash
uvicorn api:app --reload
```

The API will be available at: `http://localhost:8000`

### 3. Access Interactive Documentation

Open your browser and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📊 Disposition Categories

The API uses real NASA KOI dispositions mapped to three categories:

| Category | Description | is_exoplanet |
|----------|-------------|--------------|
| **CONFIRMED** | Confirmed exoplanet | `true` |
| **FALSE_POSITIVE** | Not a planet (stellar activity, etc.) | `false` |
| **CANDIDATE** | Potential planet, needs more study | `null` |

---

## 🔗 API Endpoints

### 1. Root - API Information
```
GET /
```

**Response:**
```json
{
  "name": "Kepler KOI Exoplanet Classification API",
  "version": "2.0.0",
  "model_loaded": true,
  "total_kois": 59316,
  "confirmed_count": 52498,
  "false_positive_count": 4839,
  "candidate_count": 1979,
  "endpoints": { ... }
}
```

---

### 2. List All KOIs
```
GET /planets
```

**Query Parameters:**
- `skip` (int): Pagination offset (default: 0)
- `limit` (int): Max results (1-1000, default: 100)
- `disposition` (str): Filter by CONFIRMED, FALSE_POSITIVE, or CANDIDATE
- `only_confirmed` (bool): Return only confirmed exoplanets
- `include_prediction` (bool): Include ML predictions (default: false)

**Example Requests:**

```bash
# Get first 10 KOIs
curl "http://localhost:8000/planets?limit=10"

# Get only confirmed exoplanets
curl "http://localhost:8000/planets?disposition=CONFIRMED&limit=50"

# Get candidates with ML predictions
curl "http://localhost:8000/planets?disposition=CANDIDATE&include_prediction=true&limit=20"

# Get only confirmed exoplanets (alternative)
curl "http://localhost:8000/planets?only_confirmed=true&limit=100"
```

**Response:**
```json
[
  {
    "kepid": 10797460,
    "kepoi_name": "K00752.01",
    "kepler_name": "Kepler-227 b",
    "x_pc": 269.96,
    "y_pc": -670.40,
    "z_pc": 806.66,
    "dist_ly": 3532.47,
    "is_exoplanet": true,
    "disposition": "CONFIRMED",
    "disposition_source": "dataset",
    "model_prediction": {
      "CANDIDATE": 0.0234,
      "CONFIRMED": 0.9456,
      "FALSE POSITIVE": 0.0310
    }
  },
  ...
]
```

---

### 3. Get Specific KOI by ID
```
GET /planets/{kepid}
```

**Path Parameters:**
- `kepid` (int): Kepler ID

**Query Parameters:**
- `include_prediction` (bool): Include ML prediction (default: false)

**Example Requests:**

```bash
# Get KOI without prediction
curl "http://localhost:8000/planets/10797460"

# Get KOI with ML prediction
curl "http://localhost:8000/planets/10797460?include_prediction=true"
```

**Response:**
```json
{
  "kepid": 10797460,
  "kepoi_name": "K00752.01",
  "kepler_name": "Kepler-227 b",
  "x_pc": 269.96,
  "y_pc": -670.40,
  "z_pc": 806.66,
  "dist_ly": 3532.47,
  "is_exoplanet": true,
  "disposition": "CONFIRMED",
  "disposition_source": "dataset",
  "model_prediction": {
    "CANDIDATE": 0.0234,
    "CONFIRMED": 0.9456,
    "FALSE POSITIVE": 0.0310
  }
}
```

---

### 4. Get ML Prediction for KOI
```
GET /planets/{kepid}/predict
```

**Path Parameters:**
- `kepid` (int): Kepler ID

**Example Request:**

```bash
curl "http://localhost:8000/planets/10797460/predict"
```

**Response:**
```json
{
  "kepid": 10797460,
  "kepoi_name": "K00752.01",
  "actual_disposition": "CONFIRMED",
  "predicted_disposition": "CONFIRMED",
  "confidence": 0.9456,
  "probabilities": {
    "CANDIDATE": 0.0234,
    "CONFIRMED": 0.9456,
    "FALSE POSITIVE": 0.0310
  }
}
```

---

### 5. Model Status
```
GET /model/status
```

**Response:**
```json
{
  "model_loaded": true,
  "model_available": true,
  "features_count": 115,
  "classes": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
}
```

---

### 6. Dataset Statistics
```
GET /stats
```

**Response:**
```json
{
  "total_kois": 59316,
  "disposition_counts": {
    "CONFIRMED": 52498,
    "FALSE_POSITIVE": 4839,
    "CANDIDATE": 1979
  },
  "disposition_percentages": {
    "CONFIRMED": 88.51,
    "FALSE_POSITIVE": 8.16,
    "CANDIDATE": 3.34
  },
  "with_coordinates": 55230,
  "with_distance": 53686,
  "model_available": true
}
```

---

## 🐍 Python Client Examples

### Example 1: Get Confirmed Exoplanets

```python
import requests

# Get first 100 confirmed exoplanets
response = requests.get(
    "http://localhost:8000/planets",
    params={
        "disposition": "CONFIRMED",
        "limit": 100
    }
)

exoplanets = response.json()
print(f"Found {len(exoplanets)} confirmed exoplanets")

for planet in exoplanets[:5]:
    print(f"{planet['kepler_name']}: {planet['dist_ly']:.2f} light years away")
```

### Example 2: Get Predictions for Candidates

```python
import requests

# Get candidates with ML predictions
response = requests.get(
    "http://localhost:8000/planets",
    params={
        "disposition": "CANDIDATE",
        "include_prediction": True,
        "limit": 50
    }
)

candidates = response.json()

for candidate in candidates:
    pred = candidate['model_prediction']
    best_pred = max(pred, key=pred.get)
    confidence = pred[best_pred]
    
    print(f"{candidate['kepoi_name']}:")
    print(f"  Actual: CANDIDATE")
    print(f"  Predicted: {best_pred} ({confidence:.2%} confidence)")
```

### Example 3: Predict Single KOI

```python
import requests

kepid = 10797460

# Get prediction
response = requests.get(f"http://localhost:8000/planets/{kepid}/predict")
result = response.json()

print(f"KOI: {result['kepoi_name']}")
print(f"Actual Disposition: {result['actual_disposition']}")
print(f"Predicted Disposition: {result['predicted_disposition']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"\nAll Probabilities:")
for class_name, prob in result['probabilities'].items():
    print(f"  {class_name}: {prob:.4f}")
```

### Example 4: Get Statistics

```python
import requests

response = requests.get("http://localhost:8000/stats")
stats = response.json()

print("Dataset Statistics:")
print(f"  Total KOIs: {stats['total_kois']:,}")
print(f"\nDisposition Breakdown:")
for disp, count in stats['disposition_counts'].items():
    pct = stats['disposition_percentages'][disp]
    print(f"  {disp}: {count:,} ({pct}%)")
print(f"\nModel Available: {stats['model_available']}")
```

---

## 🧪 Testing the API

### Using cURL

```bash
# Test API is running
curl http://localhost:8000/

# Get first 5 KOIs
curl "http://localhost:8000/planets?limit=5"

# Get a specific KOI with prediction
curl "http://localhost:8000/planets/10797460?include_prediction=true"

# Get model status
curl http://localhost:8000/model/status

# Get statistics
curl http://localhost:8000/stats
```

### Using Python requests

```python
import requests

BASE_URL = "http://localhost:8000"

# Test all endpoints
print("1. API Info:")
print(requests.get(f"{BASE_URL}/").json())

print("\n2. First 5 KOIs:")
print(requests.get(f"{BASE_URL}/planets?limit=5").json())

print("\n3. Specific KOI:")
print(requests.get(f"{BASE_URL}/planets/10797460").json())

print("\n4. Model Status:")
print(requests.get(f"{BASE_URL}/model/status").json())

print("\n5. Statistics:")
print(requests.get(f"{BASE_URL}/stats").json())
```

---

## 🔧 Configuration

### Running on Different Port

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

### Production Deployment

```bash
# Install production server
pip install gunicorn

# Run with multiple workers
gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## 📋 Response Fields Explained

| Field | Type | Description |
|-------|------|-------------|
| `kepid` | int | Kepler ID (unique identifier) |
| `kepoi_name` | str | KOI designation (e.g., K00752.01) |
| `kepler_name` | str | Kepler name if confirmed (e.g., Kepler-227 b) |
| `x_pc` | float | X coordinate in parsecs (Sun-centric) |
| `y_pc` | float | Y coordinate in parsecs |
| `z_pc` | float | Z coordinate in parsecs |
| `dist_ly` | float | Distance in light years |
| `is_exoplanet` | bool | True=confirmed, False=false positive, null=candidate |
| `disposition` | str | CONFIRMED, FALSE_POSITIVE, or CANDIDATE |
| `disposition_source` | str | Source of disposition ("dataset" or "model") |
| `model_prediction` | dict | ML prediction probabilities (if requested) |

---

## 🚨 Error Handling

### Common Errors

**404 - Not Found**
```json
{
  "detail": "Kepler ID 99999999 not found in table."
}
```

**503 - Service Unavailable (Model Not Loaded)**
```json
{
  "detail": "ML model not loaded. Train the model first using train_koi_disposition.py"
}
```

**422 - Validation Error**
```json
{
  "detail": [
    {
      "loc": ["query", "limit"],
      "msg": "ensure this value is less than or equal to 1000",
      "type": "value_error.number.not_le"
    }
  ]
}
```

---

## 🔄 Model Integration

### How It Works

1. **Startup**: API automatically loads the most recent trained model from `model_outputs/`
2. **Predictions**: When `include_prediction=true`, the API:
   - Extracts features from the KOI data
   - Preprocesses using the saved scaler
   - Runs prediction through XGBoost model
   - Returns probabilities for each class

### Model Requirements

The API requires these files in `model_outputs/`:
- `xgboost_koi_disposition_TIMESTAMP.joblib` - Trained model
- `scaler_TIMESTAMP.joblib` - Feature scaler
- `label_encoder_TIMESTAMP.joblib` - Label encoder
- `feature_names_TIMESTAMP.json` - Feature list

### Without a Trained Model

If no model is available:
- API still works for basic data access
- `model_prediction` will be `null`
- `/planets/{kepid}/predict` endpoint returns 503 error
- `model_loaded` status will be `false`

To train the model:
```bash
python train_koi_disposition.py
```

---

## 💡 Use Cases

### 1. **Exploring Confirmed Exoplanets**
```bash
curl "http://localhost:8000/planets?only_confirmed=true&limit=100"
```

### 2. **Analyzing Candidates**
Get all candidates with ML predictions to see which might be confirmed:
```bash
curl "http://localhost:8000/planets?disposition=CANDIDATE&include_prediction=true"
```

### 3. **Building Visualizations**
Fetch data with coordinates for 3D visualization:
```python
import requests
import pandas as pd

response = requests.get("http://localhost:8000/planets?limit=1000")
df = pd.DataFrame(response.json())
# Now visualize using matplotlib, plotly, etc.
```

### 4. **Model Validation**
Compare actual vs predicted dispositions:
```python
response = requests.get(
    "http://localhost:8000/planets",
    params={"include_prediction": True, "limit": 500}
)
data = response.json()

correct = 0
total = 0
for item in data:
    if item['model_prediction']:
        pred = max(item['model_prediction'], key=item['model_prediction'].get)
        if pred == item['disposition']:
            correct += 1
        total += 1

accuracy = correct / total if total > 0 else 0
print(f"Model Accuracy: {accuracy:.2%}")
```

---

## 📚 Additional Resources

- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Training Guide**: See `TRAINING_GUIDE.md` for model details

---

## 🎯 Next Steps

1. ✅ Run `python train_koi_disposition.py` to train the model
2. ✅ Start the API with `uvicorn api:app --reload`
3. ✅ Visit http://localhost:8000/docs for interactive testing
4. ✅ Integrate with your application
5. ✅ Deploy to production

---

**Happy exoplanet hunting! 🚀🔭**

