# main.py
from pathlib import Path
from typing import List, Optional, Dict
import glob
import warnings

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import joblib
import json

warnings.filterwarnings('ignore')

DATA_PATH = Path("data/koi_with_relative_location.csv")
MODEL_DIR = Path("model_outputs")

# ---------- load & prepare the table once at startup ----------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found – run fetch.py first to create it."
    )

df = pd.read_csv(DATA_PATH)

# Check if required column exists
if "koi_disposition" not in df.columns:
    print(f"⚠️  WARNING: 'koi_disposition' column not found in data file!")
    print(f"   Available columns: {list(df.columns[:10])}...")
    print(f"   Please run fetch.py to regenerate the data file.")
    raise ValueError(
        "Data file is missing 'koi_disposition' column. "
        "Run 'python fetch.py' to fetch data with all required columns."
    )

# Map dispositions to simplified categories
def map_disposition(disp):
    """Map disposition to boolean and category."""
    if pd.isna(disp):
        return None, "UNKNOWN"
    disp_upper = str(disp).upper()
    if "CONFIRMED" in disp_upper:
        return True, "CONFIRMED"
    elif "FALSE" in disp_upper:
        return False, "FALSE_POSITIVE"
    elif "CANDIDATE" in disp_upper:
        return None, "CANDIDATE"
    else:
        return None, "UNKNOWN"

# Store actual disposition for reference
df["actual_disposition"] = df["koi_disposition"].apply(lambda x: map_disposition(x)[1])
df["actual_is_exoplanet"] = df["koi_disposition"].apply(lambda x: map_disposition(x)[0])

# ---------- Load trained model (if available) ----------
trained_model = None
scaler = None
label_encoder = None
feature_names = None

def load_latest_model():
    """Load the most recent trained model artifacts."""
    global trained_model, scaler, label_encoder, feature_names
    
    try:
        if not MODEL_DIR.exists():
            print(f"⚠️  Model directory {MODEL_DIR} not found. Model predictions will be unavailable.")
            return False
        
        # Find latest model files
        model_files = list(MODEL_DIR.glob("xgboost_koi_disposition_*.joblib"))
        if not model_files:
            print("⚠️  No trained model found. Run train_koi_disposition.py first.")
            return False
        
        # Get most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        # Extract timestamp: xgboost_koi_disposition_20251004_192622.joblib -> 20251004_192622
        timestamp = "_".join(latest_model.stem.split("_")[-2:])
        
        # Load artifacts
        trained_model = joblib.load(latest_model)
        scaler = joblib.load(MODEL_DIR / f"scaler_{timestamp}.joblib")
        label_encoder = joblib.load(MODEL_DIR / f"label_encoder_{timestamp}.joblib")
        
        with open(MODEL_DIR / f"feature_names_{timestamp}.json", 'r') as f:
            feature_names = json.load(f)
        
        print(f"✅ Loaded model from {latest_model.name}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Classes: {label_encoder.classes_}")
        return True
        
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return False

# Try to load model at startup
model_loaded = load_latest_model()

# ---------- Predict dispositions for all KOIs ----------
def predict_all_dispositions():
    """Run model predictions on all KOIs and add to dataframe."""
    global df
    
    if not model_loaded or trained_model is None:
        print("⚠️  Model not loaded. Using actual dispositions as fallback.")
        df["disposition"] = df["actual_disposition"]
        df["is_exoplanet"] = df["actual_is_exoplanet"]
        df["disposition_source"] = "dataset"
        df["prediction_confidence"] = None
        return
    
    print(f"🔮 Predicting dispositions for {len(df)} KOIs...")
    
    try:
        # Prepare features for all rows
        X_all = df.copy()
        
        # Select features
        available_features = [f for f in feature_names if f in X_all.columns]
        X_features = pd.DataFrame(columns=feature_names)
        
        for feat in available_features:
            X_features[feat] = X_all[feat]
        
        # Fill missing features with 0
        X_features = X_features.fillna(0)
        
        # Scale
        X_scaled = scaler.transform(X_features)
        
        # Predict
        predictions = trained_model.predict(X_scaled)
        probabilities = trained_model.predict_proba(X_scaled)
        
        # Decode predictions
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        # Get confidence (max probability)
        confidences = probabilities.max(axis=1)
        
        # Add to dataframe
        df["disposition"] = predicted_labels
        df["disposition_source"] = "model"
        df["prediction_confidence"] = confidences
        
        # Map to is_exoplanet
        def pred_to_bool(disp):
            if disp == "CONFIRMED":
                return True
            elif disp == "FALSE POSITIVE":
                return False
            else:
                return None
        
        df["is_exoplanet"] = df["disposition"].apply(pred_to_bool)
        
        # Store full probabilities for quick access
        for i, class_name in enumerate(label_encoder.classes_):
            df[f"prob_{class_name}"] = probabilities[:, i]
        
        print(f"✅ Predictions complete!")
        print(f"   Predicted CONFIRMED: {(df['disposition'] == 'CONFIRMED').sum()}")
        print(f"   Predicted FALSE POSITIVE: {(df['disposition'] == 'FALSE POSITIVE').sum()}")
        print(f"   Predicted CANDIDATE: {(df['disposition'] == 'CANDIDATE').sum()}")
        
        # Compare with actual
        if "actual_disposition" in df.columns:
            matches = (df["disposition"] == df["actual_disposition"]).sum()
            accuracy = matches / len(df) * 100
            print(f"   Agreement with dataset: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"⚠️  Error predicting dispositions: {e}")
        print(f"   Falling back to dataset dispositions")
        df["disposition"] = df["actual_disposition"]
        df["is_exoplanet"] = df["actual_is_exoplanet"]
        df["disposition_source"] = "dataset"
        df["prediction_confidence"] = None

# Run predictions at startup
predict_all_dispositions()

# -------- Response Models --------
class PlanetInfo(BaseModel):
    kepid: int
    kepoi_name: Optional[str]
    kepler_name: Optional[str]
    x_pc: Optional[float]
    y_pc: Optional[float]
    z_pc: Optional[float]
    dist_ly: Optional[float]
    is_exoplanet: Optional[bool]
    disposition: str
    disposition_source: str
    confidence: Optional[float] = None
    actual_disposition: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None

class ModelStatus(BaseModel):
    model_loaded: bool
    model_available: bool
    features_count: Optional[int]
    classes: Optional[List[str]]

# -------- FastAPI app -----------
app = FastAPI(
    title="Kepler KOI Exoplanet Classification API",
    description="Serves Kepler Objects of Interest with real disposition data "
                "and optional ML model predictions using XGBoost classifier.",
    version="2.0.0",
)

# helper – remap a dataframe row to the JSON we want to expose
def row_to_dict(row: pd.Series, include_actual: bool = False, include_probabilities: bool = False) -> dict:
    """Convert DataFrame row to API response dictionary."""
    
    result = {
        "kepid": int(row.kepid),
        "kepoi_name": row.kepoi_name if pd.notna(row.kepoi_name) else None,
        "kepler_name": row.kepler_name if pd.notna(row.kepler_name) else None,
        "x_pc": None if pd.isna(row.x_pc) else float(row.x_pc),
        "y_pc": None if pd.isna(row.y_pc) else float(row.y_pc),
        "z_pc": None if pd.isna(row.z_pc) else float(row.z_pc),
        "dist_ly": None if pd.isna(row.dist_ly) else float(row.dist_ly),
        "is_exoplanet": None if pd.isna(row.is_exoplanet) else bool(row.is_exoplanet),
        "disposition": row.disposition,
        "disposition_source": row.disposition_source,
        "confidence": None if pd.isna(row.prediction_confidence) else float(row.prediction_confidence),
    }
    
    # Optionally include actual disposition for comparison
    if include_actual and "actual_disposition" in row:
        result["actual_disposition"] = row.actual_disposition
    
    # Optionally include full probabilities
    if include_probabilities and model_loaded:
        probabilities = {}
        for class_name in label_encoder.classes_:
            col_name = f"prob_{class_name}"
            if col_name in row:
                probabilities[class_name] = float(row[col_name])
        if probabilities:
            result["probabilities"] = probabilities
    
    return result

# -------- endpoints --------
@app.get("/", summary="API Info")
def root():
    """
    Get API information and status.
    """
    return {
        "name": "Kepler KOI Exoplanet Classification API",
        "version": "3.0.0",
        "description": "API for Kepler Objects of Interest with ML-predicted dispositions",
        "model_loaded": model_loaded,
        "total_kois": len(df),
        "predicted_confirmed": int((df["disposition"] == "CONFIRMED").sum()),
        "predicted_false_positive": int((df["disposition"] == "FALSE POSITIVE").sum()),
        "predicted_candidate": int((df["disposition"] == "CANDIDATE").sum()),
        "prediction_source": df["disposition_source"].iloc[0] if len(df) > 0 else "unknown",
        "endpoints": {
            "/planets": "List all KOIs with ML predictions",
            "/planets/{kepid}": "Get specific KOI by Kepler ID",
            "/model/status": "Get model status and information",
            "/stats": "Get dataset statistics and model accuracy"
        }
    }

@app.get("/planets", response_model=List[dict])
def list_planets(
    skip: int = Query(0, ge=0, description="Rows to skip (pagination start)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum rows to return"),
    disposition: Optional[str] = Query(
        None, 
        description="Filter by predicted disposition: CONFIRMED, FALSE_POSITIVE, CANDIDATE"
    ),
    only_confirmed: Optional[bool] = Query(
        None, description="If true, return only predicted confirmed exoplanets"
    ),
    include_actual: bool = Query(
        False, description="Include actual disposition from dataset for comparison"
    ),
    include_probabilities: bool = Query(
        False, description="Include full probability distribution"
    ),
):
    """
    List KOI rows with ML-predicted dispositions.
    
    **Parameters:**
    - **skip**: Number of rows to skip (pagination)
    - **limit**: Maximum rows to return (1-1000)
    - **disposition**: Filter by predicted disposition
    - **only_confirmed**: Only return predicted confirmed exoplanets
    - **include_actual**: Include actual disposition for comparison
    - **include_probabilities**: Include full probabilities
    
    **Returns:**
    - List of KOI objects with ML-predicted dispositions
    """
    subset = df.copy()
    
    # Filter by predicted disposition
    if disposition:
        disp_upper = disposition.upper().replace("_", " ")  # FALSE_POSITIVE -> FALSE POSITIVE
        subset = subset[subset["disposition"] == disp_upper]
    
    # Filter predicted confirmed only
    if only_confirmed is True:
        subset = subset[subset["is_exoplanet"] == True]

    # Pagination
    rows = subset.iloc[skip : skip + limit]
    
    return [row_to_dict(r, include_actual=include_actual, include_probabilities=include_probabilities) 
            for _, r in rows.iterrows()]


@app.get("/planets/{kepid}", response_model=dict)
def get_planet(
    kepid: int,
    include_actual: bool = Query(
        False, description="Include actual disposition for comparison"
    ),
    include_probabilities: bool = Query(
        False, description="Include full probability distribution"
    )
):
    """
    Get a single KOI by its Kepler ID with ML-predicted disposition.
    
    **Parameters:**
    - **kepid**: Kepler ID (integer)
    - **include_actual**: Include actual disposition for comparison
    - **include_probabilities**: Include full probabilities
    
    **Returns:**
    - KOI object with ML-predicted disposition
    """
    rows = df[df["kepid"] == kepid]
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Kepler ID {kepid} not found in table.",
        )
    return row_to_dict(rows.iloc[0], include_actual=include_actual, include_probabilities=include_probabilities)


@app.get("/model/status", response_model=ModelStatus)
def model_status():
    """
    Get the status of the ML model.
    
    **Returns:**
    - Model loading status and configuration
    """
    return {
        "model_loaded": model_loaded,
        "model_available": model_loaded,
        "features_count": len(feature_names) if feature_names else None,
        "classes": list(label_encoder.classes_) if label_encoder else None,
    }


@app.get("/stats", response_model=dict)
def get_statistics():
    """
    Get dataset statistics, predictions, and model accuracy.
    
    **Returns:**
    - Comprehensive statistics about predictions and model performance
    """
    stats = {
        "total_kois": len(df),
        "predicted_counts": {
            "CONFIRMED": int((df["disposition"] == "CONFIRMED").sum()),
            "FALSE_POSITIVE": int((df["disposition"] == "FALSE POSITIVE").sum()),
            "CANDIDATE": int((df["disposition"] == "CANDIDATE").sum()),
        },
        "predicted_percentages": {
            "CONFIRMED": round((df["disposition"] == "CONFIRMED").sum() / len(df) * 100, 2),
            "FALSE_POSITIVE": round((df["disposition"] == "FALSE POSITIVE").sum() / len(df) * 100, 2),
            "CANDIDATE": round((df["disposition"] == "CANDIDATE").sum() / len(df) * 100, 2),
        },
        "actual_counts": {
            "CONFIRMED": int((df["actual_disposition"] == "CONFIRMED").sum()),
            "FALSE_POSITIVE": int((df["actual_disposition"] == "FALSE_POSITIVE").sum()),
            "CANDIDATE": int((df["actual_disposition"] == "CANDIDATE").sum()),
        },
        "with_coordinates": int(df[["x_pc", "y_pc", "z_pc"]].notna().all(axis=1).sum()),
        "with_distance": int(df["dist_ly"].notna().sum()),
        "model_available": model_loaded,
        "disposition_source": df["disposition_source"].iloc[0] if len(df) > 0 else "unknown",
    }
    
    # Add model accuracy if predictions are available
    if model_loaded and "actual_disposition" in df.columns:
        matches = (df["disposition"] == df["actual_disposition"]).sum()
        stats["model_accuracy"] = round(matches / len(df) * 100, 2)
        stats["correct_predictions"] = int(matches)
        
        # Per-class accuracy
        per_class_accuracy = {}
        for class_name in ["CONFIRMED", "FALSE_POSITIVE", "CANDIDATE"]:
            actual_class = df["actual_disposition"] == class_name
            if actual_class.sum() > 0:
                correct = ((df["disposition"] == df["actual_disposition"]) & actual_class).sum()
                per_class_accuracy[class_name] = round(correct / actual_class.sum() * 100, 2)
        stats["per_class_accuracy"] = per_class_accuracy
    
    return stats
