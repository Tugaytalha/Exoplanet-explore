"""
Test script to diagnose model loading issues
"""
from pathlib import Path
import joblib
import json

MODEL_DIR = Path("model_outputs")

print("="*70)
print("MODEL LOADING DIAGNOSTIC")
print("="*70)

# Check 1: Directory exists
print(f"\n1. Checking if directory exists...")
print(f"   Path: {MODEL_DIR}")
print(f"   Exists: {MODEL_DIR.exists()}")
print(f"   Is directory: {MODEL_DIR.is_dir()}")

if not MODEL_DIR.exists():
    print("\n❌ ERROR: model_outputs directory not found!")
    exit(1)

# Check 2: List files
print(f"\n2. Listing files in {MODEL_DIR}:")
files = list(MODEL_DIR.iterdir())
for f in files:
    print(f"   - {f.name}")

# Check 3: Find model files
print(f"\n3. Searching for model files...")
model_files = list(MODEL_DIR.glob("xgboost_koi_disposition_*.joblib"))
print(f"   Found {len(model_files)} model file(s)")
for f in model_files:
    print(f"   - {f.name}")
    print(f"     Size: {f.stat().st_size / 1024:.2f} KB")
    print(f"     Modified: {f.stat().st_mtime}")

if not model_files:
    print("\n❌ ERROR: No model files found!")
    exit(1)

# Check 4: Get latest model
print(f"\n4. Selecting latest model...")
latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
print(f"   Latest: {latest_model.name}")

# Extract timestamp: xgboost_koi_disposition_20251004_192622.joblib -> 20251004_192622
timestamp = "_".join(latest_model.stem.split("_")[-2:])
print(f"   Timestamp: {timestamp}")

# Check 5: Check all required files
print(f"\n5. Checking required files...")
required_files = {
    "model": MODEL_DIR / f"xgboost_koi_disposition_{timestamp}.joblib",
    "scaler": MODEL_DIR / f"scaler_{timestamp}.joblib",
    "encoder": MODEL_DIR / f"label_encoder_{timestamp}.joblib",
    "features": MODEL_DIR / f"feature_names_{timestamp}.json",
}

all_exist = True
for name, path in required_files.items():
    exists = path.exists()
    status = "✅" if exists else "❌"
    print(f"   {status} {name}: {path.name}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ ERROR: Some required files are missing!")
    exit(1)

# Check 6: Try loading each file
print(f"\n6. Attempting to load files...")

try:
    print(f"   Loading model...")
    model = joblib.load(required_files["model"])
    print(f"   ✅ Model loaded: {type(model).__name__}")
except Exception as e:
    print(f"   ❌ Failed to load model: {e}")
    exit(1)

try:
    print(f"   Loading scaler...")
    scaler = joblib.load(required_files["scaler"])
    print(f"   ✅ Scaler loaded: {type(scaler).__name__}")
except Exception as e:
    print(f"   ❌ Failed to load scaler: {e}")
    exit(1)

try:
    print(f"   Loading label encoder...")
    encoder = joblib.load(required_files["encoder"])
    print(f"   ✅ Encoder loaded: {type(encoder).__name__}")
    print(f"      Classes: {encoder.classes_}")
except Exception as e:
    print(f"   ❌ Failed to load encoder: {e}")
    exit(1)

try:
    print(f"   Loading feature names...")
    with open(required_files["features"], 'r') as f:
        features = json.load(f)
    print(f"   ✅ Features loaded: {len(features)} features")
except Exception as e:
    print(f"   ❌ Failed to load features: {e}")
    exit(1)

# Check 7: Verify model attributes
print(f"\n7. Verifying model attributes...")
print(f"   Model type: {type(model)}")
print(f"   Has predict: {hasattr(model, 'predict')}")
print(f"   Has predict_proba: {hasattr(model, 'predict_proba')}")

# Success!
print("\n" + "="*70)
print("✅ ALL CHECKS PASSED!")
print("="*70)
print(f"\nModel Details:")
print(f"  Timestamp: {timestamp}")
print(f"  Features: {len(features)}")
print(f"  Classes: {list(encoder.classes_)}")
print(f"\nThe model files are valid and loadable.")
print(f"If API still fails, check for import errors or path issues.")

