# main.py
from pathlib import Path
from typing import List, Optional, Dict
import glob
import warnings

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

warnings.filterwarnings('ignore')

DATA_PATH = Path("data/koi_with_relative_location.csv")
MODEL_DIR = Path("model_outputs")

# ---------- MongoDB Configuration ----------
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "exoplanet_db")

# MongoDB client
mongo_client = None
mongo_db = None
mongo_collection = None

def connect_to_mongodb():
    """Connect to MongoDB and return client, db, and collection."""
    global mongo_client, mongo_db, mongo_collection
    
    try:
        mongo_client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        # Test connection
        mongo_client.admin.command('ping')
        mongo_db = mongo_client[MONGODB_DB]
        mongo_collection = mongo_db['planets']
        print(f"✅ Connected to MongoDB at {MONGODB_URL}")
        print(f"   Database: {MONGODB_DB}")
        print(f"   Collection: planets")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"⚠️  Could not connect to MongoDB: {e}")
        print(f"   Continuing without MongoDB integration")
        return False
    except Exception as e:
        print(f"⚠️  Unexpected error connecting to MongoDB: {e}")
        return False

# Try to connect to MongoDB at startup
mongo_connected = connect_to_mongodb()

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
imputer = None
scaler = None
label_encoder = None
feature_names = None

def load_latest_model():
    """Load the most recent trained model artifacts."""
    global trained_model, imputer, scaler, label_encoder, feature_names
    
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
        
        # Load imputer (handle old models without imputer)
        imputer_path = MODEL_DIR / f"imputer_{timestamp}.joblib"
        if imputer_path.exists():
            imputer = joblib.load(imputer_path)
            print(f"✅ Loaded imputer")
        else:
            print(f"⚠️  No imputer found (old model). Missing values will be filled with 0.")
            imputer = None
        
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
        
        # Select only the features used in training, in the correct order
        missing_features = [f for f in feature_names if f not in X_all.columns]
        if missing_features:
            print(f"⚠️  {len(missing_features)} features missing from data, filling with 0")
            for feat in missing_features:
                X_all[feat] = 0
        
        # Select features in exact order as training
        X_features = X_all[feature_names].copy()
        
        # Impute missing values using the same strategy as training
        if imputer is not None:
            X_imputed = imputer.transform(X_features)
        else:
            # Fallback for old models without imputer
            print("⚠️  Using zero-fill (no imputer available)")
            X_imputed = X_features.fillna(0).values
        
        # Scale
        X_scaled = scaler.transform(X_imputed)
        
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

# ---------- Write predictions to MongoDB ----------
def populate_mongodb():
    """Write all predicted data to MongoDB."""
    if not mongo_connected or mongo_collection is None:
        print("⚠️  MongoDB not connected. Skipping data population.")
        return False
    
    print("📤 Writing data to MongoDB...")
    
    try:
        # Convert DataFrame to list of dicts
        records = df.to_dict('records')
        
        # Clean up records (convert NaN to None, handle numpy types)
        cleaned_records = []
        for record in records:
            cleaned_record = {}
            for key, value in record.items():
                # Convert numpy types to Python types
                if pd.isna(value):
                    cleaned_record[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    cleaned_record[key] = value.item()
                elif isinstance(value, np.bool_):
                    cleaned_record[key] = bool(value)
                else:
                    cleaned_record[key] = value
            cleaned_records.append(cleaned_record)
        
        # Drop existing collection and recreate
        mongo_collection.drop()
        print(f"   Inserting {len(cleaned_records)} documents...")
        
        # Insert in batches for better performance
        batch_size = 1000
        for i in range(0, len(cleaned_records), batch_size):
            batch = cleaned_records[i:i + batch_size]
            mongo_collection.insert_many(batch, ordered=False)
        
        # Create index on kepid for fast lookups
        mongo_collection.create_index('kepid', unique=True)
        mongo_collection.create_index('disposition')
        mongo_collection.create_index('is_exoplanet')
        
        print(f"✅ Successfully wrote {len(cleaned_records)} documents to MongoDB")
        print(f"   Created indexes on: kepid, disposition, is_exoplanet")
        return True
        
    except Exception as e:
        print(f"❌ Error writing to MongoDB: {e}")
        return False

# Populate MongoDB with predicted data
mongodb_populated = populate_mongodb()

# Optimize DataFrame for faster queries (fallback if MongoDB fails)
if not mongodb_populated:
    print("🔧 Optimizing DataFrame for queries...")
    df.set_index('kepid', drop=False, inplace=True)
    print("✅ DataFrame optimized (using as fallback)")

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
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

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
    
    if include_probabilities and model_loaded:
        prob_columns = [f"prob_{class_name}" for class_name in label_encoder.classes_]
        columns_to_include.extend([col for col in prob_columns if col in df_subset.columns])
    
    # Use pandas to_dict with 'records' orientation for fast conversion
    result_df = df_subset[columns_to_include].copy()
    
    # Rename prediction_confidence to confidence
    if 'prediction_confidence' in result_df.columns:
        result_df.rename(columns={'prediction_confidence': 'confidence'}, inplace=True)
    
    # Convert to dict efficiently
    records = result_df.to_dict('records')
    
    # Post-process for probabilities structure (if needed)
    if include_probabilities and model_loaded:
        prob_cols = [f"prob_{class_name}" for class_name in label_encoder.classes_]
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
    # Get counts from MongoDB if available, otherwise from DataFrame
    if mongodb_populated and mongo_collection is not None:
        try:
            total_kois = mongo_collection.count_documents({})
            predicted_confirmed = mongo_collection.count_documents({"disposition": "CONFIRMED"})
            predicted_false_positive = mongo_collection.count_documents({"disposition": "FALSE POSITIVE"})
            predicted_candidate = mongo_collection.count_documents({"disposition": "CANDIDATE"})
            data_source = "mongodb"
        except:
            total_kois = len(df)
            predicted_confirmed = int((df["disposition"] == "CONFIRMED").sum())
            predicted_false_positive = int((df["disposition"] == "FALSE POSITIVE").sum())
            predicted_candidate = int((df["disposition"] == "CANDIDATE").sum())
            data_source = "dataframe"
    else:
        total_kois = len(df)
        predicted_confirmed = int((df["disposition"] == "CONFIRMED").sum())
        predicted_false_positive = int((df["disposition"] == "FALSE POSITIVE").sum())
        predicted_candidate = int((df["disposition"] == "CANDIDATE").sum())
        data_source = "dataframe"
    
    return {
        "name": "Kepler KOI Exoplanet Classification API",
        "version": "3.0.0",
        "description": "API for Kepler Objects of Interest with ML-predicted dispositions",
        "model_loaded": model_loaded,
        "mongodb_connected": mongo_connected,
        "mongodb_populated": mongodb_populated,
        "data_source": data_source,
        "total_kois": total_kois,
        "predicted_confirmed": predicted_confirmed,
        "predicted_false_positive": predicted_false_positive,
        "predicted_candidate": predicted_candidate,
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
    
    # Use MongoDB if available and populated
    if mongodb_populated and mongo_collection is not None:
        try:
            # Build MongoDB query
            query = {}
            
            # Filter by disposition
            if disposition:
                disp_upper = disposition.upper().replace("_", " ")  # FALSE_POSITIVE -> FALSE POSITIVE
                query['disposition'] = disp_upper
            
            # Filter by confirmed only
            if only_confirmed is True:
                query['is_exoplanet'] = True
            
            # Build projection (fields to include/exclude)
            projection = {'_id': 0}  # Exclude MongoDB _id
            
            # Exclude fields based on parameters
            if not include_actual:
                projection['actual_disposition'] = 0
                projection['actual_is_exoplanet'] = 0
            
            if not include_probabilities:
                # Exclude probability fields
                projection['prob_CANDIDATE'] = 0
                projection['prob_CONFIRMED'] = 0
                projection['prob_FALSE POSITIVE'] = 0
            
            # Query MongoDB with pagination
            cursor = mongo_collection.find(query, projection).skip(skip)
            
            if limit is not None:
                cursor = cursor.limit(limit)
            
            # Convert cursor to list
            results = list(cursor)
            
            print(f"📊 MongoDB query returned {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Error querying MongoDB: {e}")
            print(f"   Falling back to DataFrame...")
            # Fall through to DataFrame fallback
    
    # Fallback to DataFrame operations
    print("📊 Using DataFrame (MongoDB not available)")
    subset = df.copy()
    
    # Filter by predicted disposition
    if disposition:
        disp_upper = disposition.upper().replace("_", " ")  # FALSE_POSITIVE -> FALSE POSITIVE
        subset = subset[subset["disposition"] == disp_upper]
    
    # Filter predicted confirmed only
    if only_confirmed is True:
        subset = subset[subset["is_exoplanet"] == True]

    # Pagination
    if limit is None:
        # Return all rows from skip onwards
        rows = subset.iloc[skip:]
    else:
        # Return limited rows
    rows = subset.iloc[skip : skip + limit]
    
    # Use optimized vectorized conversion
    results = df_to_dict_list(rows, include_actual=include_actual, include_probabilities=include_probabilities)
    return results


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
    
    # Use MongoDB if available and populated
    if mongodb_populated and mongo_collection is not None:
        try:
            # Build projection
            projection = {'_id': 0}
            
            if not include_actual:
                projection['actual_disposition'] = 0
                projection['actual_is_exoplanet'] = 0
            
            if not include_probabilities:
                projection['prob_CANDIDATE'] = 0
                projection['prob_CONFIRMED'] = 0
                projection['prob_FALSE POSITIVE'] = 0
            
            # Query MongoDB
            planet_data = mongo_collection.find_one({'kepid': kepid}, projection)
            
            if planet_data:
                return planet_data
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Kepler ID {kepid} not found in database.",
                )
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Error querying MongoDB for kepid {kepid}: {e}")
            print(f"   Falling back to DataFrame...")
            # Fall through to DataFrame fallback
    
    # Fallback to DataFrame
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
