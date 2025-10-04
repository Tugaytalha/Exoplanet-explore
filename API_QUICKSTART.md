# 🚀 API Quick Start

## What Changed?

Your API now returns **REAL exoplanet disposition data** instead of dummy flags!

### Before (v1.0) ❌
```json
{
  "is_exoplanet": true  // Dummy flag based on distance
}
```

### After (v2.0) ✅
```json
{
  "is_exoplanet": true,
  "disposition": "CONFIRMED",
  "disposition_source": "dataset",
  "model_prediction": {
    "CANDIDATE": 0.023,
    "CONFIRMED": 0.946,
    "FALSE POSITIVE": 0.031
  }
}
```

---

## 🎯 Three Classification Categories

| Category | Meaning | is_exoplanet |
|----------|---------|--------------|
| **CONFIRMED** | Confirmed exoplanet ✅ | `true` |
| **FALSE_POSITIVE** | Not a planet ❌ | `false` |
| **CANDIDATE** | Needs more study ❓ | `null` |

---

## ⚡ Quick Commands

### 1. Start the API
```bash
uvicorn api:app --reload
```
Visit: http://localhost:8000/docs

### 2. Test It
```bash
# Get API info
curl http://localhost:8000/

# Get first 5 KOIs
curl "http://localhost:8000/planets?limit=5"

# Get confirmed exoplanets only
curl "http://localhost:8000/planets?disposition=CONFIRMED&limit=10"

# Get specific KOI with ML prediction
curl "http://localhost:8000/planets/10797460?include_prediction=true"

# Get just the prediction
curl "http://localhost:8000/planets/10797460/predict"
```

---

## 🆕 New Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | API info and stats |
| `GET /planets` | List KOIs (with filtering) |
| `GET /planets/{kepid}` | Get specific KOI |
| `GET /planets/{kepid}/predict` | Get ML prediction |
| `GET /model/status` | Check if model is loaded |
| `GET /stats` | Dataset statistics |

---

## 🔥 New Features

### 1. Filter by Disposition
```bash
# Get only confirmed exoplanets
curl "http://localhost:8000/planets?disposition=CONFIRMED&limit=50"

# Get candidates
curl "http://localhost:8000/planets?disposition=CANDIDATE&limit=50"

# Get false positives
curl "http://localhost:8000/planets?disposition=FALSE_POSITIVE&limit=50"
```

### 2. ML Model Predictions
```bash
# Include predictions in list
curl "http://localhost:8000/planets?include_prediction=true&limit=10"

# Get prediction for specific KOI
curl "http://localhost:8000/planets/10797460/predict"
```

### 3. Statistics
```bash
curl http://localhost:8000/stats
```

**Response:**
```json
{
  "total_kois": 59316,
  "confirmed_count": 52498,
  "false_positive_count": 4839,
  "candidate_count": 1979,
  "model_available": true
}
```

---

## 🐍 Python Examples

### Get Confirmed Exoplanets
```python
import requests

response = requests.get(
    "http://localhost:8000/planets",
    params={"disposition": "CONFIRMED", "limit": 100}
)
exoplanets = response.json()

for planet in exoplanets[:5]:
    print(f"{planet['kepler_name']}: {planet['dist_ly']:.1f} ly away")
```

### Get ML Predictions
```python
import requests

response = requests.get("http://localhost:8000/planets/10797460/predict")
result = response.json()

print(f"KOI: {result['kepoi_name']}")
print(f"Actual: {result['actual_disposition']}")
print(f"Predicted: {result['predicted_disposition']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## ⚙️ Setup Requirements

### 1. Install New Dependencies
```bash
pip install joblib
```

### 2. Train the Model (Optional but Recommended)
```bash
python train_koi_disposition.py
```

**Note**: The API works without a trained model, but predictions won't be available.

### 3. Run the API
```bash
uvicorn api:app --reload
```

---

## 📊 Response Example

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

## 🎯 What This Means for Your Project

### ✅ Real Data
- No more dummy flags
- Actual NASA KOI dispositions
- Three-category classification system

### ✅ ML Integration
- XGBoost model predictions
- Confidence scores
- Compare actual vs predicted

### ✅ Better Filtering
- Filter by disposition type
- Get only confirmed exoplanets
- Analyze candidates

### ✅ Production Ready
- Proper error handling
- Model status monitoring
- Comprehensive statistics

---

## 📚 Full Documentation

For complete details, see:
- **API_GUIDE.md** - Complete API documentation
- **TRAINING_GUIDE.md** - Model training guide
- **INSTRUCTIONS.md** - Competition guide

---

## 🔍 Quick Test

```python
import requests

# Test the API
response = requests.get("http://localhost:8000/")
info = response.json()

print(f"API Version: {info['version']}")
print(f"Model Loaded: {info['model_loaded']}")
print(f"Total KOIs: {info['total_kois']:,}")
print(f"Confirmed: {info['confirmed_count']:,}")
print(f"False Positives: {info['false_positive_count']:,}")
print(f"Candidates: {info['candidate_count']:,}")
```

---

**Your API is now ready with real exoplanet data! 🌟🚀**

