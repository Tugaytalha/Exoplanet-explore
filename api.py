# main.py
from pathlib import Path
from typing import List, Optional, Dict
import glob
import warnings

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import ORJSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import joblib
import json

warnings.filterwarnings('ignore')

# Try to import orjson for faster JSON serialization
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    print("⚠️  orjson not available, using standard JSON (slower)")

# Try to import Polars for faster DataFrame operations
try:
    import polars as pl
    USE_POLARS = True
    print("✅ Using Polars for ultra-fast DataFrame operations")
except ImportError:
    USE_POLARS = False
    print("⚠️  Polars not available, using pandas (slower). Install with: pip install polars")

DATA_PATH = Path("data/koi_with_relative_location.csv")
MODEL_DIR = Path("model_outputs")

# ---------- load & prepare the table once at startup ----------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"{DATA_PATH} not found – run fetch.py first to create it."
    )

# Load data with Polars (faster) or Pandas (fallback)
if USE_POLARS:
    print("📊 Loading data with Polars...")
    try:
        # Specify problematic columns as strings to avoid parsing errors
        # koi_quarters contains binary strings like "11111111111111111000000000000000"
        schema_overrides = {
            'koi_quarters': pl.Utf8,  # Treat as string
            'koi_limbdark_mod': pl.Utf8,  # Might be text
            'koi_trans_mod': pl.Utf8,  # Might be text
            'koi_sparprov': pl.Utf8,  # Might be text
            'koi_fittype': pl.Utf8,  # Might be text
            'koi_comment': pl.Utf8,  # Text
            'ra_str': pl.Utf8,  # Text
            'dec_str': pl.Utf8,  # Text
            'rastr': pl.Utf8,  # Text
            'decstr': pl.Utf8,  # Text
        }
        
        df_polars = pl.read_csv(
            DATA_PATH,
            infer_schema_length=10000,  # Infer schema from more rows
            schema_overrides=schema_overrides,  # Force string types for problematic columns
            ignore_errors=False
        )
        # Convert to pandas for compatibility with existing model code
        df = df_polars.to_pandas()
        print(f"✅ Data loaded with Polars: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"⚠️  Polars failed: {str(e)[:200]}")
        print("   Falling back to Pandas...")
        USE_POLARS = False
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Data loaded with Pandas: {len(df)} rows")
else:
    print("📊 Loading data with Pandas...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded with Pandas: {len(df)} rows")

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

# Optimize DataFrame for faster queries
print("🔧 Optimizing DataFrame for queries...")

# 1. Convert string columns to categorical for memory efficiency
categorical_columns = ['disposition', 'disposition_source', 'actual_disposition', 
                       'kepoi_name', 'kepler_name']
for col in categorical_columns:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = df[col].astype('category')

# 2. Downcast numeric columns to save memory
float_cols = df.select_dtypes(include=['float64']).columns
for col in float_cols:
    df[col] = pd.to_numeric(df[col], downcast='float')

# 3. Set kepid as index for O(1) lookups
df.set_index('kepid', drop=False, inplace=True)

# 4. Report memory usage
memory_usage_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"✅ DataFrame optimized! Memory usage: {memory_usage_mb:.2f} MB")
print(f"   - Categorical columns: {len(categorical_columns)}")
print(f"   - Indexed by kepid for fast lookups")

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
    version="3.0.0",
    default_response_class=ORJSONResponse if USE_ORJSON else None,  # Use faster JSON serialization
)

# Add GZip compression middleware for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

print("✅ FastAPI optimizations enabled:")
if USE_ORJSON:
    print("   - orjson: Fast JSON serialization")
print("   - gzip: Response compression for data > 1KB")

# helper – convert dataframe to list of dictionaries efficiently
def df_to_dict_list(df_subset: pd.DataFrame, include_actual: bool = False, include_probabilities: bool = False) -> list:
    """Convert DataFrame to list of API response dictionaries using vectorized operations."""
    
    # Select base columns
    base_columns = [
        'kepid', 'kepoi_name', 'kepler_name', 'x_pc', 'y_pc', 'z_pc', 
        'dist_ly', 'is_exoplanet', 'disposition', 'disposition_source', 
        'prediction_confidence'
    ]
    
    # Build column list dynamically based on what's available
    columns_to_include = [col for col in base_columns if col in df_subset.columns]
    
    if include_actual and 'actual_disposition' in df_subset.columns:
        columns_to_include.append('actual_disposition')
    
    prob_col_names = []
    if include_probabilities and model_loaded:
        prob_col_names = [f"prob_{class_name}" for class_name in label_encoder.classes_]
        columns_to_include.extend([col for col in prob_col_names if col in df_subset.columns])
    
    # Use pandas to_dict with 'records' orientation for fast conversion
    # Reset index to avoid index in output
    result_df = df_subset[columns_to_include].reset_index(drop=True)
    
    # Rename prediction_confidence to confidence
    if 'prediction_confidence' in result_df.columns:
        result_df.rename(columns={'prediction_confidence': 'confidence'}, inplace=True)
    
    # Convert to dict efficiently using orient='records' (fastest for this use case)
    records = result_df.to_dict('records')
    
    # Post-process for probabilities structure (if needed) - optimized
    if include_probabilities and model_loaded and prob_col_names:
        for record in records:
            probabilities = {}
            for class_name in label_encoder.classes_:
                col_name = f"prob_{class_name}"
                if col_name in record:
                    probabilities[class_name] = record.pop(col_name)
            if probabilities:
                record['probabilities'] = probabilities
    
    return records

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
async def list_planets(
    skip: int = Query(0, ge=0, description="Rows to skip (pagination start)"),
    limit: Optional[int] = Query(100, ge=1, description="Maximum rows to return (default: 100, no maximum limit)"),
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
    - **limit**: Maximum rows to return (default: 100, set to None or very high value for all results)
    - **disposition**: Filter by predicted disposition
    - **only_confirmed**: Only return predicted confirmed exoplanets
    - **include_actual**: Include actual disposition for comparison
    - **include_probabilities**: Include full probabilities
    
    **Returns:**
    - List of KOI objects with ML-predicted dispositions
    """
    # Use view instead of copy for better performance
    subset = df
    
    # Apply filters early to reduce dataset size (optimized order)
    # Filter by disposition first (most selective)
    if disposition:
        disp_upper = disposition.upper().replace("_", " ")  # FALSE_POSITIVE -> FALSE POSITIVE
        # Use categorical comparison for speed
        subset = subset[subset["disposition"] == disp_upper]
    
    # Then filter by confirmed status
    if only_confirmed is True:
        subset = subset[subset["is_exoplanet"] == True]

    # Apply pagination on filtered data
    if limit is None:
        rows = subset.iloc[skip:]
    else:
        rows = subset.iloc[skip : skip + limit]
    
    # Use optimized vectorized conversion
    return df_to_dict_list(rows, include_actual=include_actual, include_probabilities=include_probabilities)


@app.get("/planets/{kepid}", response_model=dict)
async def get_planet(
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
    # Use index-based lookup for O(1) access
    try:
        row = df.loc[kepid:kepid]  # Returns DataFrame with single row
        if row.empty:
            raise KeyError
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Kepler ID {kepid} not found in table.",
        )
    
    # Use optimized conversion
    result = df_to_dict_list(row, include_actual=include_actual, include_probabilities=include_probabilities)
    return result[0]


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
