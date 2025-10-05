# main.py
from pathlib import Path
from typing import List, Optional, Dict
import glob
import warnings

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import joblib
import json
import os
import io
import base64
from datetime import datetime
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
def load_data_from_mongodb():
    """Load data from MongoDB if available and populated."""
    if not mongo_connected or mongo_collection is None:
        return None
    
    try:
        # Check if MongoDB has data
        doc_count = mongo_collection.count_documents({})
        
        if doc_count == 0:
            print("📊 MongoDB is empty, will load from CSV")
            return None
        
        print(f"📊 Loading {doc_count} documents from MongoDB...")
        
        # Load all documents from MongoDB (excluding _id)
        # Load in batches for better memory efficiency with large datasets
        batch_size = 10000
        documents = []
        
        cursor = mongo_collection.find({}, {'_id': 0})
        batch = []
        for doc in cursor:
            batch.append(doc)
            if len(batch) >= batch_size:
                documents.extend(batch)
                batch = []
                if len(documents) % 50000 == 0:
                    print(f"   Loaded {len(documents)} documents...")
        
        # Add remaining documents
        if batch:
            documents.extend(batch)
        
        # Convert to DataFrame
        df = pd.DataFrame(documents)
        
        print(f"✅ Loaded {len(df)} rows from MongoDB")
        return df
        
    except Exception as e:
        print(f"⚠️  Error loading from MongoDB: {e}")
        return None

# Try to load from MongoDB first
print("\n" + "="*70)
print("DATA LOADING")
print("="*70)

df = load_data_from_mongodb()

# If MongoDB doesn't have data, load from CSV
if df is None:
    print("📊 Loading data from CSV...")
            
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found – run fetch.py first to create it."
        )

df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} rows from CSV")
    
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

# Check if data already has predictions (loaded from MongoDB with predictions)
data_has_predictions = (
    'disposition' in df.columns and 
    'disposition_source' in df.columns and 
    'prediction_confidence' in df.columns and
    df['disposition_source'].notna().any()
)

print("\n" + "="*70)
print("DATA PREPARATION")
print("="*70)

if data_has_predictions:
    print("✅ Data already contains predictions (loaded from MongoDB)")
    print(f"   Source: {df['disposition_source'].iloc[0] if len(df) > 0 else 'unknown'}")
    print(f"   Predicted CONFIRMED: {(df['disposition'] == 'CONFIRMED').sum()}")
    print(f"   Predicted FALSE POSITIVE: {(df['disposition'] == 'FALSE POSITIVE').sum()}")
    print(f"   Predicted CANDIDATE: {(df['disposition'] == 'CANDIDATE').sum()}")
    
    # Ensure actual disposition columns exist for comparison
    if 'actual_disposition' not in df.columns and 'koi_disposition' in df.columns:
        df["actual_disposition"] = df["koi_disposition"].apply(lambda x: map_disposition(x)[1])
    if 'actual_is_exoplanet' not in df.columns and 'koi_disposition' in df.columns:
        df["actual_is_exoplanet"] = df["koi_disposition"].apply(lambda x: map_disposition(x)[0])
else:
    # Fresh data from CSV - need to create actual disposition columns
    if 'koi_disposition' in df.columns:
        df["actual_disposition"] = df["koi_disposition"].apply(lambda x: map_disposition(x)[1])
        df["actual_is_exoplanet"] = df["koi_disposition"].apply(lambda x: map_disposition(x)[0])
        print("📊 Prepared actual disposition columns from CSV data")
        print(f"   Dataset CONFIRMED: {(df['actual_disposition'] == 'CONFIRMED').sum()}")
        print(f"   Dataset FALSE_POSITIVE: {(df['actual_disposition'] == 'FALSE_POSITIVE').sum()}")
        print(f"   Dataset CANDIDATE: {(df['actual_disposition'] == 'CANDIDATE').sum()}")

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
    
    # Check if predictions already exist
    if data_has_predictions:
        print("⏭️  Skipping predictions - data already contains model predictions")
        return
    
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

# ---------- Update MongoDB with predictions ----------
def update_mongodb_predictions():
    """Update MongoDB with prediction results only."""
    if not mongo_connected or mongo_collection is None:
        print("⚠️  MongoDB not connected. Skipping MongoDB update.")
        return False
    
    # Check if MongoDB already has predictions
    try:
        existing_count = mongo_collection.count_documents({})
        if existing_count > 0 and data_has_predictions:
            print(f"✅ MongoDB already contains {existing_count} documents with predictions")
            return True
    except:
        pass
    
    # Check if we have predictions to write
    if not data_has_predictions and 'disposition' not in df.columns:
        print("⚠️  No predictions to write to MongoDB")
        return False
    
    try:
        existing_count = mongo_collection.count_documents({})
        
        if existing_count == 0:
            # MongoDB is empty, need to insert all data
            print("📤 MongoDB is empty - inserting all data with predictions...")
            
            # Convert DataFrame to list of dicts
            records = df.to_dict('records')
            
            # Clean up records (convert NaN to None, handle numpy types)
            cleaned_records = []
            for record in records:
                cleaned_record = {}
                for key, value in record.items():
                    if pd.isna(value):
                        cleaned_record[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        cleaned_record[key] = value.item()
                    elif isinstance(value, np.bool_):
                        cleaned_record[key] = bool(value)
                    else:
                        cleaned_record[key] = value
                cleaned_records.append(cleaned_record)
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(cleaned_records), batch_size):
                batch = cleaned_records[i:i + batch_size]
                mongo_collection.insert_many(batch, ordered=False)
                if (i + batch_size) % 10000 == 0:
                    print(f"   Inserted {i + batch_size} documents...")
            
            # Create indexes
            mongo_collection.create_index('kepid', unique=True)
            mongo_collection.create_index('disposition')
            mongo_collection.create_index('is_exoplanet')
            
            print(f"✅ Successfully inserted {len(cleaned_records)} documents to MongoDB")
            print(f"   Created indexes on: kepid, disposition, is_exoplanet")
            return True
        else:
            # MongoDB has data, update only prediction fields
            print(f"📤 Updating {len(df)} documents with predictions in MongoDB...")
            
            # Prediction fields to update
            prediction_fields = [
                'disposition', 'disposition_source', 'prediction_confidence', 
                'is_exoplanet'
            ]
            
            # Add probability columns if they exist
            if model_loaded and label_encoder is not None:
                for class_name in label_encoder.classes_:
                    prob_col = f"prob_{class_name}"
                    if prob_col in df.columns:
                        prediction_fields.append(prob_col)
            
            # Update in batches
            update_count = 0
            batch_size = 1000
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                
                for _, row in batch_df.iterrows():
                    kepid = int(row['kepid'])
                    
                    # Build update document
                    update_doc = {}
                    for field in prediction_fields:
                        if field in row.index:
                            value = row[field]
                            if pd.isna(value):
                                update_doc[field] = None
                            elif isinstance(value, (np.integer, np.floating)):
                                update_doc[field] = value.item()
                            elif isinstance(value, np.bool_):
                                update_doc[field] = bool(value)
                            else:
                                update_doc[field] = value
                    
                    # Update MongoDB document
                    mongo_collection.update_one(
                        {'kepid': kepid},
                        {'$set': update_doc}
                    )
                    update_count += 1
                
                if (i + batch_size) % 10000 == 0:
                    print(f"   Updated {i + batch_size} documents...")
            
            print(f"✅ Successfully updated {update_count} documents in MongoDB with predictions")
            return True
        
    except Exception as e:
        print(f"❌ Error updating MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return False

# Update MongoDB with predictions
mongodb_populated = update_mongodb_predictions()

# Optimize DataFrame for faster queries (fallback if MongoDB fails)
if not mongodb_populated:
    print("🔧 Optimizing DataFrame for queries...")
    df.set_index('kepid', drop=False, inplace=True)
    print("✅ DataFrame optimized (using as fallback)")

# -------- Response Models --------
class PlanetInfo(BaseModel):
    """
    Comprehensive KOI response model with all available fields.
    
    **Categories:**
    - Identity/Location: kepid, tic_id, kepoi_name, kepler_name, hostname, ra, dec
    - Transit Core: koi_period, koi_time0bk, koi_duration, koi_depth, koi_ror, koi_dor, koi_impact, koi_model_snr, koi_num_transits
    - Stellar: st_teff, st_rad, st_mass, koi_steff, koi_srad, koi_smass
    - Derived: koi_prad, koi_sma, koi_teq, koi_insol
    - Disposition: koi_disposition, koi_pdisposition, disposition, is_exoplanet, koi_fpflag_*
    - Photometry: koi_kepmag, sy_gaiamag, sy_tmag
    - 3D/Sky: sy_dist, sy_plx, x_pc, y_pc, z_pc, dist_ly
    
    All fields are optional and depend on data availability.
    """
    # Identity
    kepid: int
    tic_id: Optional[str] = None
    kepoi_name: Optional[str] = None
    kepler_name: Optional[str] = None
    hostname: Optional[str] = None
    
    # Coordinates
    ra: Optional[float] = None
    dec: Optional[float] = None
    ra_str: Optional[str] = None
    dec_str: Optional[str] = None
    
    # Transit parameters (core)
    koi_period: Optional[float] = None
    koi_period_err1: Optional[float] = None
    koi_period_err2: Optional[float] = None
    koi_time0bk: Optional[float] = None
    koi_duration: Optional[float] = None
    koi_depth: Optional[float] = None
    koi_ror: Optional[float] = None
    koi_dor: Optional[float] = None
    koi_impact: Optional[float] = None
    koi_model_snr: Optional[float] = None
    koi_num_transits: Optional[int] = None
    
    # Stellar parameters
    st_teff: Optional[float] = None
    st_rad: Optional[float] = None
    st_mass: Optional[float] = None
    koi_steff: Optional[float] = None
    koi_srad: Optional[float] = None
    koi_smass: Optional[float] = None
    
    # Derived parameters
    koi_prad: Optional[float] = None
    koi_sma: Optional[float] = None
    koi_teq: Optional[float] = None
    koi_insol: Optional[float] = None
    
    # Disposition
    koi_disposition: Optional[str] = None
    koi_pdisposition: Optional[str] = None
    disposition: str
    disposition_source: str
    is_exoplanet: Optional[bool] = None
    confidence: Optional[float] = None
    
    # False positive flags
    koi_fpflag_nt: Optional[int] = None
    koi_fpflag_ss: Optional[int] = None
    koi_fpflag_co: Optional[int] = None
    koi_fpflag_ec: Optional[int] = None
    
    # Photometry
    koi_kepmag: Optional[float] = None
    sy_gaiamag: Optional[float] = None
    sy_tmag: Optional[float] = None
    
    # 3D position
    sy_dist: Optional[float] = None
    sy_plx: Optional[float] = None
    x_pc: Optional[float] = None
    y_pc: Optional[float] = None
    z_pc: Optional[float] = None
    dist_ly: Optional[float] = None
    
    # Optional fields
    actual_disposition: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    
    class Config:
        extra = "allow"  # Allow additional fields from MongoDB

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
    
    # Essential columns - Kimlik/konum (Identity/Location)
    base_columns = [
        'kepid', 'tic_id', 'kepoi_name', 'kepler_name', 'hostname',
        'ra', 'dec', 'ra_str', 'dec_str',
    ]
    
    # Transit core columns - Transit çekirdeği
    transit_columns = [
        'koi_period', 'koi_period_err1', 'koi_period_err2',
        'koi_time0bk', 'koi_time0bk_err1', 'koi_time0bk_err2',
        'koi_duration', 'koi_duration_err1', 'koi_duration_err2',
        'koi_depth', 'koi_depth_err1', 'koi_depth_err2',
        'koi_ror', 'koi_ror_err1', 'koi_ror_err2',
        'koi_dor', 'koi_dor_err1', 'koi_dor_err2',
        'koi_impact', 'koi_impact_err1', 'koi_impact_err2',
        'koi_model_snr', 'koi_num_transits',
    ]
    
    # Stellar parameters - Yıldız
    stellar_columns = [
        'st_teff', 'st_rad', 'st_mass',
        'koi_steff', 'koi_srad', 'koi_smass',
        'koi_steff_err1', 'koi_steff_err2',
        'koi_srad_err1', 'koi_srad_err2',
        'koi_smass_err1', 'koi_smass_err2',
    ]
    
    # Derived/display parameters - Türev/gösterim
    derived_columns = [
        'koi_prad', 'koi_prad_err1', 'koi_prad_err2',
        'koi_sma', 'koi_sma_err1', 'koi_sma_err2',
        'koi_teq', 'koi_teq_err1', 'koi_teq_err2',
        'koi_insol', 'koi_insol_err1', 'koi_insol_err2',
    ]
    
    # Disposition/filters - Rozet/filtre
    disposition_columns = [
        'koi_disposition', 'koi_pdisposition',
        'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
        'disposition', 'disposition_source', 'prediction_confidence',
        'is_exoplanet',
    ]
    
    # Photometry - Fotometri (minimal set only)
    photometry_columns = [
        'koi_kepmag', 'koi_kepmag_err',
        'sy_gaiamag', 'sy_gaiamagerr1',
        'sy_tmag', 'sy_tmagerr1',
    ]
    
    # 3D/Sky position - 3D/sky (minimal set only)
    sky_columns = [
        'sy_dist', 'sy_disterr1', 'sy_disterr2',
        'sy_plx', 'sy_plxerr1', 'sy_plxerr2',
        'x_pc', 'y_pc', 'z_pc', 'dist_ly',
    ]
    
    # Explicitly exclude these columns (metadata and unwanted fields)
    excluded_columns = [
        # Vetting/delivery metadata
        'koi_delivname', 'koi_vet_stat', 'koi_quarters',
        # Diagnostic/TCE statistics
        'koi_count', 'koi_max_sngle_ev', 'koi_max_mult_ev', 'koi_bin_oedp_sig',
        # Limb darkening
        'koi_limbdark_mod', 'koi_ldm_coeff1', 'koi_ldm_coeff2', 'koi_ldm_coeff3', 'koi_ldm_coeff4',
        # Model fitting metadata
        'koi_trans_mod', 'koi_model_dof', 'koi_model_chisq',
        # Orbital parameters not listed
        'koi_eccen', 'koi_eccen_err1', 'koi_eccen_err2',
        'koi_longp', 'koi_longp_err1', 'koi_longp_err2',
        # Alternate epoch fields
        'koi_time0', 'koi_time0_err1', 'koi_time0_err2',
        # Ingress duration
        'koi_ingress', 'koi_ingress_err1', 'koi_ingress_err2',
        # Inclination
        'koi_incl', 'koi_incl_err1', 'koi_incl_err2',
        # Provenance/commentary
        'koi_sparprov', 'koi_comment', 'koi_vet_date', 'koi_tce_plnt_num',
        'koi_tce_delivname', 'koi_disp_prov', 'koi_parm_prov',
        # Extra identifiers
        'host', 'hd_name', 'hip_name',
        # Coordinate transforms
        'glon', 'glat', 'elon', 'elat', 'ra_str', 'dec_str', 'rastr', 'decstr',
        # Unwanted st_* stellar fields (keep only st_teff, st_rad, st_mass)
        'st_met', 'st_meterr1', 'st_meterr2',
        'st_logg', 'st_loggerr1', 'st_loggerr2',
        'st_age', 'st_ageerr1', 'st_ageerr2',
        'st_lum', 'st_lumerr1', 'st_lumerr2',
        'st_dens', 'st_denserr1', 'st_denserr2',
        'st_radv', 'st_radverr1', 'st_radverr2',
        'st_vsin', 'st_vsinerr1', 'st_vsinerr2',
        'st_rotp', 'st_rotperr1', 'st_rotperr2',
        # Unwanted sy_* fields (keep only sy_gaiamag, sy_tmag, sy_dist, sy_plx)
        'sy_icmag', 'sy_icmagerr1', 'sy_icmagerr2',
        'sy_bmag', 'sy_bmagerr1', 'sy_bmagerr2',
        'sy_vmag', 'sy_vmagerr1', 'sy_vmagerr2',
        'sy_umag', 'sy_umagerr1', 'sy_umagerr2',
        'sy_rmag', 'sy_rmagerr1', 'sy_rmagerr2',
        'sy_imag', 'sy_imagerr1', 'sy_imagerr2',
        'sy_zmag', 'sy_zmagerr1', 'sy_zmagerr2',
        'sy_jmag', 'sy_jmagerr1', 'sy_jmagerr2',
        'sy_hmag', 'sy_hmagerr1', 'sy_hmagerr2',
        'sy_kmag', 'sy_kmagerr1', 'sy_kmagerr2',
        'sy_w1mag', 'sy_w1magerr1', 'sy_w1magerr2',
        'sy_w2mag', 'sy_w2magerr1', 'sy_w2magerr2',
        'sy_w3mag', 'sy_w3magerr1', 'sy_w3magerr2',
        'sy_w4mag', 'sy_w4magerr1', 'sy_w4magerr2',
        # Proper motion/astrometric (not wanted)
        'sy_pm', 'sy_pmerr1', 'sy_pmerr2',
        'sy_pmra', 'sy_pmraerr1', 'sy_pmraerr2',
        'sy_pmdec', 'sy_pmdecerr1', 'sy_pmdecerr2',
        # Other sy_* fields not wanted
        'sy_kepmag', 'sy_kepmagerr1', 'sy_kepmagerr2',
        'sy_pnum',
        # Additional unwanted fields
        'kepoi_id',
    ]
    
    # Combine wanted columns
    all_columns = (base_columns + transit_columns + stellar_columns + 
                   derived_columns + disposition_columns + 
                   photometry_columns + sky_columns)
    
    # Build column list: include only wanted columns that exist and aren't excluded
    columns_to_include = [
        col for col in all_columns 
        if col in df_subset.columns and col not in excluded_columns
    ]
    
    if include_actual and 'actual_disposition' in df_subset.columns:
        if 'actual_disposition' not in columns_to_include:
            columns_to_include.append('actual_disposition')
        if 'actual_is_exoplanet' in df_subset.columns and 'actual_is_exoplanet' not in columns_to_include:
            columns_to_include.append('actual_is_exoplanet')
    
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
            "/api/features": "Get list of required features for prediction",
            "/api/predict": "Predict disposition from feature values (POST)",
            "/api/train": "Train custom model with uploaded data (POST)",
            "/api/latest-visualization": "Get latest training visualization as base64",
            "/api/download/{filename}": "Download trained model files",
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
    List KOI rows with ML-predicted dispositions and comprehensive planet data.
    
    **Parameters:**
    - **skip**: Number of rows to skip (pagination)
    - **limit**: Maximum rows to return (default: 100, set to None or very high value for all results)
    - **disposition**: Filter by predicted disposition (CONFIRMED, FALSE_POSITIVE, CANDIDATE)
    - **only_confirmed**: Only return predicted confirmed exoplanets
    - **include_actual**: Include actual disposition for comparison
    - **include_probabilities**: Include full probability distribution
    
    **Returns:**
    - List of KOI objects with comprehensive data including:
      - **Identity/Location**: kepid, tic_id, ra, dec, kepoi_name, kepler_name, hostname
      - **Transit Core**: koi_period, koi_time0bk, koi_duration, koi_depth, koi_ror, koi_dor, koi_impact, koi_model_snr, koi_num_transits (+ error columns)
      - **Stellar Parameters**: st_teff, st_rad, st_mass, koi_steff, koi_srad, koi_smass (+ error columns)
      - **Derived Parameters**: koi_prad, koi_sma, koi_teq, koi_insol (+ error columns)
      - **Disposition/Filters**: koi_disposition, koi_pdisposition, disposition, is_exoplanet, koi_fpflag_*
      - **Photometry**: koi_kepmag, sy_gaiamag, sy_tmag (+ error columns)
      - **3D/Sky Position**: sy_dist, sy_plx, x_pc, y_pc, z_pc, dist_ly (+ error columns)
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
            # By default, include all fields except _id and conditionally exclude some
            projection = {'_id': 0}  # Exclude MongoDB _id
            
            # Conditionally exclude fields based on parameters
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
    Get a single KOI by its Kepler ID with comprehensive planet data.
    
    **Parameters:**
    - **kepid**: Kepler ID (integer)
    - **include_actual**: Include actual disposition for comparison
    - **include_probabilities**: Include full probability distribution
    
    **Returns:**
    - KOI object with comprehensive data including all transit, stellar, derived, 
      photometric, and positional parameters (see /planets endpoint for full list)
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


@app.get("/api/features")
def get_required_features():
    """
    Get the list of features required for prediction.
    
    **Returns:**
    - List of all feature names
    - Count of features
    - Model status
    """
    
    if not model_loaded or not feature_names:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Feature list not available."
        )
    
    return {
        "features": feature_names,
        "count": len(feature_names),
        "model_loaded": model_loaded,
        "classes": list(label_encoder.classes_) if label_encoder else None
    }


@app.post("/api/predict")
def predict_disposition(features: dict):
    """
    Predict KOI disposition from input features.
    
    **Parameters:**
    - **features**: Dictionary of feature values (must match training feature names)
    
    **Returns:**
    - Predicted disposition class
    - Probability distribution across all classes
    - Confidence score
    
    **Example request body:**
    ```json
    {
        "koi_period": 15.85567,
        "koi_duration": 2.8933,
        "koi_depth": 944.0,
        "koi_prad": 2.89,
        "st_teff": 5455,
        "st_rad": 0.927,
        ...
    }
    ```
    """
    
    if not model_loaded or trained_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Cannot make predictions."
        )
    
    if not feature_names:
        raise HTTPException(
            status_code=503,
            detail="Feature names not available."
        )
    
    try:
        # Create feature vector in correct order
        feature_vector = []
        missing_features = []
        
        for feature_name in feature_names:
            if feature_name in features:
                value = features[feature_name]
                # Handle None/null values
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    feature_vector.append(0.0)  # Will be imputed
                else:
                    feature_vector.append(float(value))
            else:
                # Feature not provided - will be imputed
                feature_vector.append(0.0)
                missing_features.append(feature_name)
        
        # Convert to numpy array and reshape
        X = np.array(feature_vector).reshape(1, -1)
        
        # Impute missing values using the same strategy as training
        if imputer is not None:
            X = imputer.transform(X)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = trained_model.predict(X_scaled)[0]
        probabilities = trained_model.predict_proba(X_scaled)[0]
        
        # Decode prediction
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence (max probability)
        confidence = float(probabilities.max())
        
        # Build probability distribution
        prob_distribution = {}
        for i, class_name in enumerate(label_encoder.classes_):
            prob_distribution[class_name] = float(probabilities[i])
        
        # Map to is_exoplanet
        is_exoplanet = None
        if predicted_label == "CONFIRMED":
            is_exoplanet = True
        elif predicted_label == "FALSE POSITIVE":
            is_exoplanet = False
        
        return {
            "disposition": predicted_label,
            "is_exoplanet": is_exoplanet,
            "confidence": confidence,
            "probabilities": prob_distribution,
            "features_provided": len(features),
            "features_required": len(feature_names),
            "features_missing": len(missing_features),
            "missing_features_sample": missing_features[:10] if missing_features else []
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error making prediction: {str(e)}"
        )


@app.post("/api/train")
async def train_custom_model(
    training_file: UploadFile = File(..., description="CSV file with training data"),
    use_existing_data: bool = Form(default=True, description="Combine with existing data (True) or use only uploaded data (False)"),
    model_name: str = Form(default="custom_model", description="Name for the trained model")
):
    """
    Train a new XGBoost model with custom data.
    
    **Parameters:**
    - **training_file**: CSV file with training data (must include 'koi_disposition' column)
    - **use_existing_data**: If True, combines uploaded data with existing dataset. If False, uses only uploaded data.
    - **model_name**: Custom name for the trained model
    
    **Returns:**
    - Training metrics (accuracy, precision, recall, F1-score)
    - Cross-validation scores
    - Feature importance (top 20)
    - Confusion matrix
    - Model file path for download
    - Visualization plots (base64 encoded)
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/train" \\
      -F "training_file=@my_data.csv" \\
      -F "use_existing_data=true" \\
      -F "model_name=my_custom_model"
    ```
    """
    
    try:
        # Read uploaded file
        contents = await training_file.read()
        user_df = pd.read_csv(io.BytesIO(contents))
        
        print(f"\n{'='*70}")
        print(f"CUSTOM MODEL TRAINING: {model_name}")
        print(f"{'='*70}")
        print(f"📤 Uploaded data: {len(user_df)} rows, {len(user_df.columns)} columns")
        
        # Validate required column
        if 'koi_disposition' not in user_df.columns:
            raise HTTPException(
                status_code=400,
                detail="Training data must include 'koi_disposition' column"
            )
        
        # Combine with existing data if requested
        if use_existing_data:
            print(f"📊 Combining with existing data ({len(df)} rows)...")
            training_df = pd.concat([df, user_df], ignore_index=True)
            print(f"✅ Combined dataset: {len(training_df)} rows")
        else:
            print(f"📊 Using only uploaded data...")
            training_df = user_df.copy()
        
        # Import training classes
        from train_koi_disposition import KOIDispositionPredictor
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Configure training
        config = {
            'data_path': None,  # We're passing data directly
            'target_column': 'koi_disposition',
            'output_dir': 'model_outputs',
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            'model_name': f'xgboost_{model_name}',
        }
        
        # Initialize predictor
        predictor = KOIDispositionPredictor(config)
        
        # Set the data directly
        predictor.df = training_df
        
        # Run preprocessing
        print("\n📋 Preprocessing data...")
        X, y = predictor.preprocess_data()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=config['test_size'], 
            random_state=config['random_state'], stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=config['val_size']/(1-config['test_size']),
            random_state=config['random_state'], stratify=y_temp
        )
        
        print(f"✅ Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Feature engineering
        print("\n🔧 Feature engineering...")
        X_train_scaled, X_val_scaled, X_test_scaled = predictor.prepare_features(
            X_train, X_val, X_test
        )
        
        # Train model
        print("\n🚀 Training model...")
        y_val_encoded = predictor.train_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Cross-validation
        print("\n📊 Cross-validation...")
        cv_results = predictor.cross_validate_model(X, y)
        
        # Evaluate on test set
        print("\n📈 Evaluating on test set...")
        y_test_encoded = predictor.label_encoder.transform(y_test)
        y_pred = predictor.model.predict(X_test_scaled)
        y_pred_proba = predictor.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        test_accuracy = accuracy_score(y_test_encoded, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_encoded, y_pred, average='weighted'
        )
        
        # Classification report
        report = classification_report(
            y_test_encoded, y_pred,
            target_names=predictor.label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # Feature importance
        feature_importance_df = predictor.get_feature_importance()
        top_features = feature_importance_df.head(20).to_dict('records')
        
        # Generate plots
        print("\n📊 Generating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=predictor.label_encoder.classes_,
                   yticklabels=predictor.label_encoder.classes_,
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Feature Importance
        top_20_features = feature_importance_df.head(20)
        axes[0, 1].barh(range(len(top_20_features)), top_20_features['importance'])
        axes[0, 1].set_yticks(range(len(top_20_features)))
        axes[0, 1].set_yticklabels(top_20_features['feature'])
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Top 20 Feature Importance')
        
        # 3. Class Distribution
        class_counts = pd.Series(y).value_counts()
        axes[1, 0].bar(class_counts.index, class_counts.values)
        axes[1, 0].set_title('Training Data Class Distribution')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Cross-Validation Scores
        cv_scores = cv_results['fold_scores']
        axes[1, 1].plot(range(1, len(cv_scores) + 1), cv_scores, marker='o')
        axes[1, 1].axhline(y=cv_results['mean_score'], color='r', linestyle='--', 
                          label=f"Mean: {cv_results['mean_score']:.4f}")
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Cross-Validation Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Save model
        print("\n💾 Saving model...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = predictor.save_artifacts(timestamp)
        
        # Model download path
        model_filename = saved_paths['model'].split('/')[-1]
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"CV Mean: {cv_results['mean_score']:.4f}")
        
        return {
            "status": "success",
            "model_name": model_name,
            "timestamp": timestamp,
            "data_info": {
                "total_samples": len(training_df),
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "test_samples": len(X_test),
                "features_used": len(predictor.feature_names),
                "classes": list(predictor.label_encoder.classes_),
                "used_existing_data": use_existing_data
            },
            "metrics": {
                "test_accuracy": float(test_accuracy),
                "test_precision": float(precision),
                "test_recall": float(recall),
                "test_f1": float(f1),
                "cv_mean": float(cv_results['mean_score']),
                "cv_std": float(cv_results['std_score']),
                "cv_scores": [float(s) for s in cv_results['fold_scores']]
            },
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "feature_importance": top_features,
            "model_files": {
                "model": model_filename,
                "download_url": f"/api/download/{model_filename}",
                "scaler": saved_paths['scaler'].split('/')[-1],
                "label_encoder": saved_paths['label_encoder'].split('/')[-1],
                "feature_names": saved_paths['feature_names'].split('/')[-1]
            },
            "visualization": {
                "plot_base64": plot_base64,
                "description": "Base64 encoded PNG image with 4 plots: confusion matrix, feature importance, class distribution, CV scores"
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )


@app.get("/api/download/{filename}")
def download_model_file(filename: str):
    """
    Download trained model files.
    
    **Parameters:**
    - **filename**: Name of the file to download
    
    **Returns:**
    - File download
    """
    
    file_path = MODEL_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File {filename} not found"
        )
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get("/api/latest-visualization")
def get_latest_training_visualization():
    """
    Get the latest training visualization plot as base64 encoded PNG with evaluation metrics.
    
    **Returns:**
    - Base64 encoded PNG image of the latest training evaluation plots
    - Complete evaluation metrics (accuracy, precision, recall, F1)
    - Feature importance (top 20)
    - Confusion matrix
    - Cross-validation scores
    - Timestamp and file information
    
    **Response:**
    ```json
    {
      "plot_base64": "iVBORw0KGgoAAAANSUhEUgAAA...",
      "timestamp": "20251004_221246",
      "filename": "model_evaluation_20251004_221245.png",
      "created_at": "2025-10-04T22:12:45",
      "metrics": {
        "test_accuracy": 0.9816,
        "test_precision": 0.9829,
        "test_recall": 0.9816,
        "test_f1": 0.9821,
        "cv_mean": 0.9844,
        "cv_std": 0.0010,
        "cv_scores": [0.9843, 0.9850, 0.9859, 0.9835, 0.9831]
      },
      "classification_report": {...},
      "confusion_matrix": [[321, 0, 75], [12, 10484, 4], [127, 0, 841]],
      "feature_importance": [
        {"feature": "sy_pmdec", "importance": 0.2115},
        ...
      ]
    }
    ```
    
    **Example:**
    ```bash
    # Get latest visualization with metrics
    curl "http://localhost:8000/api/latest-visualization"
    
    # Extract just metrics
    curl "http://localhost:8000/api/latest-visualization" | jq '.metrics'
    
    # Display plot in HTML
    curl "http://localhost:8000/api/latest-visualization" | jq -r '.plot_base64' | \
      sed 's/^/<img src="data:image\/png;base64,/' | sed 's/$/" \/>/'
    ```
    """
    
    try:
        # Find all visualization PNG files
        viz_files = list(MODEL_DIR.glob("model_evaluation_*.png"))
        
        if not viz_files:
            raise HTTPException(
                status_code=404,
                detail="No training visualizations found. Train a model first."
            )
        
        # Get the most recent file
        latest_viz = max(viz_files, key=lambda x: x.stat().st_mtime)
        
        # Extract timestamp from filename: model_evaluation_20251004_221245.png
        filename = latest_viz.name
        # The timestamp in the PNG filename is slightly different (ends in 45 vs 46)
        # We need to find the corresponding metrics file
        timestamp_base = filename.replace("model_evaluation_", "").replace(".png", "")
        
        # Try to find matching metrics file (timestamp might differ by 1-2 seconds)
        metrics_files = list(MODEL_DIR.glob(f"evaluation_metrics_{timestamp_base[:9]}*.json"))
        if not metrics_files:
            # Fallback: find the most recent metrics file
            metrics_files = list(MODEL_DIR.glob("evaluation_metrics_*.json"))
        
        # Read and encode the image
        with open(latest_viz, 'rb') as f:
            image_data = f.read()
            plot_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Get file creation time
        created_timestamp = latest_viz.stat().st_mtime
        created_at = datetime.fromtimestamp(created_timestamp).isoformat()
        
        # Build base response
        response = {
            "plot_base64": plot_base64,
            "timestamp": timestamp_base,
            "filename": filename,
            "description": "Training visualization with confusion matrix, feature importance, class distribution, and CV scores",
            "created_at": created_at,
            "file_size_bytes": len(image_data),
            "image_format": "png"
        }
        
        # Load evaluation metrics if available
        if metrics_files:
            try:
                # Get the most recent metrics file
                latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
                metrics_timestamp = latest_metrics.stem.replace("evaluation_metrics_", "")
                
                with open(latest_metrics, 'r') as f:
                    metrics_data = json.load(f)
                
                # Extract metrics
                response["metrics"] = {
                    "test_accuracy": float(metrics_data.get("test_accuracy", 0)),
                    "test_precision": float(metrics_data.get("classification_report", {}).get("weighted avg", {}).get("precision", 0)),
                    "test_recall": float(metrics_data.get("classification_report", {}).get("weighted avg", {}).get("recall", 0)),
                    "test_f1": float(metrics_data.get("classification_report", {}).get("weighted avg", {}).get("f1-score", 0)),
                    "cv_mean": float(metrics_data.get("cv_mean", 0)),
                    "cv_std": float(metrics_data.get("cv_std", 0)),
                    "cv_scores": [float(s) for s in metrics_data.get("cv_scores", [])]
                }
                
                # Add classification report
                response["classification_report"] = metrics_data.get("classification_report", {})
                
                # Add confusion matrix
                response["confusion_matrix"] = metrics_data.get("confusion_matrix", [])
                
                # Update timestamp to match metrics file
                response["metrics_timestamp"] = metrics_timestamp
                
            except Exception as e:
                print(f"⚠️  Could not load metrics: {e}")
                response["metrics_error"] = str(e)
        
        # Load feature importance if available
        feature_importance_files = list(MODEL_DIR.glob(f"feature_importance_{timestamp_base[:9]}*.csv"))
        if not feature_importance_files:
            feature_importance_files = list(MODEL_DIR.glob("feature_importance_*.csv"))
        
        if feature_importance_files:
            try:
                latest_fi = max(feature_importance_files, key=lambda x: x.stat().st_mtime)
                fi_df = pd.read_csv(latest_fi)
                
                # Get top 20 features
                top_20 = fi_df.head(20).to_dict('records')
                response["feature_importance"] = top_20
                response["feature_importance_count"] = len(fi_df)
                
            except Exception as e:
                print(f"⚠️  Could not load feature importance: {e}")
                response["feature_importance_error"] = str(e)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error loading visualization: {str(e)}"
        )


@app.get("/stats", response_model=dict)
def get_statistics():
    """
    Get dataset statistics, predictions, and model accuracy.
    
    **Returns:**
    - Comprehensive statistics about predictions and model performance
    """
    
    # Use MongoDB for stats if available
    if mongodb_populated and mongo_collection is not None:
        try:
            total_kois = mongo_collection.count_documents({})
            
            # Predicted counts
            predicted_counts = {
                "CONFIRMED": mongo_collection.count_documents({"disposition": "CONFIRMED"}),
                "FALSE_POSITIVE": mongo_collection.count_documents({"disposition": "FALSE POSITIVE"}),
                "CANDIDATE": mongo_collection.count_documents({"disposition": "CANDIDATE"}),
            }
            
            # Actual counts
            actual_counts = {
                "CONFIRMED": mongo_collection.count_documents({"actual_disposition": "CONFIRMED"}),
                "FALSE_POSITIVE": mongo_collection.count_documents({"actual_disposition": "FALSE_POSITIVE"}),
                "CANDIDATE": mongo_collection.count_documents({"actual_disposition": "CANDIDATE"}),
            }
            
            # Calculate percentages
            predicted_percentages = {
                "CONFIRMED": round(predicted_counts["CONFIRMED"] / total_kois * 100, 2) if total_kois > 0 else 0,
                "FALSE_POSITIVE": round(predicted_counts["FALSE_POSITIVE"] / total_kois * 100, 2) if total_kois > 0 else 0,
                "CANDIDATE": round(predicted_counts["CANDIDATE"] / total_kois * 100, 2) if total_kois > 0 else 0,
            }
            
            # Count documents with coordinates and distance
            with_coordinates = mongo_collection.count_documents({
                "x_pc": {"$ne": None},
                "y_pc": {"$ne": None},
                "z_pc": {"$ne": None}
            })
            with_distance = mongo_collection.count_documents({"dist_ly": {"$ne": None}})
            
            # Get disposition source
            sample_doc = mongo_collection.find_one({}, {"disposition_source": 1})
            disposition_source = sample_doc.get("disposition_source", "unknown") if sample_doc else "unknown"
            
            stats = {
                "total_kois": total_kois,
                "predicted_counts": predicted_counts,
                "predicted_percentages": predicted_percentages,
                "actual_counts": actual_counts,
                "with_coordinates": with_coordinates,
                "with_distance": with_distance,
                "model_available": model_loaded,
                "mongodb_source": True,
                "disposition_source": disposition_source,
            }
            
            # Calculate model accuracy using MongoDB aggregation
            if model_loaded:
                # Count where disposition == actual_disposition
                correct_predictions = mongo_collection.count_documents({
                    "$expr": {"$eq": ["$disposition", "$actual_disposition"]}
                })
                
                stats["model_accuracy"] = round(correct_predictions / total_kois * 100, 2) if total_kois > 0 else 0
                stats["correct_predictions"] = correct_predictions
                
                # Per-class accuracy
                per_class_accuracy = {}
                for class_name in ["CONFIRMED", "FALSE_POSITIVE", "CANDIDATE"]:
                    actual_count = actual_counts[class_name]
                    if actual_count > 0:
                        correct = mongo_collection.count_documents({
                            "actual_disposition": class_name,
                            "$expr": {"$eq": ["$disposition", "$actual_disposition"]}
                        })
                        per_class_accuracy[class_name] = round(correct / actual_count * 100, 2)
                stats["per_class_accuracy"] = per_class_accuracy
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting stats from MongoDB: {e}")
            print(f"   Falling back to DataFrame...")
            # Fall through to DataFrame fallback
    
    # Fallback to DataFrame
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
        "mongodb_source": False,
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
