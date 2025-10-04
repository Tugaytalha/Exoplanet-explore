"""
Example Usage Script
====================
This script demonstrates how to:
1. Train the model
2. Load a trained model
3. Make predictions on new data
4. Analyze results
"""

import pandas as pd
import joblib
import json

# ============================================================================
# EXAMPLE 1: Train the model
# ============================================================================

def train_model_example():
    """
    Example: Training the model
    """
    print("="*70)
    print("EXAMPLE 1: Training the Model")
    print("="*70)
    print("\nTo train the model, simply run:")
    print("  python train_koi_disposition.py")
    print("\nThis will:")
    print("  - Load and preprocess the data")
    print("  - Train an XGBoost classifier")
    print("  - Perform cross-validation")
    print("  - Evaluate on test set")
    print("  - Save all artifacts to model_outputs/")
    print("\nExpected runtime: 5-15 minutes")


# ============================================================================
# EXAMPLE 2: Load trained model and inspect
# ============================================================================

def inspect_model_example():
    """
    Example: Load and inspect a trained model
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Loading and Inspecting a Trained Model")
    print("="*70)
    
    # These are example paths - replace with your actual file paths
    model_path = "model_outputs/xgboost_koi_disposition_TIMESTAMP.joblib"
    features_path = "model_outputs/feature_names_TIMESTAMP.json"
    importance_path = "model_outputs/feature_importance_TIMESTAMP.csv"
    
    print(f"\nTo load a trained model:")
    print(f"  model = joblib.load('{model_path}')")
    
    print(f"\nTo view feature names:")
    print(f"  with open('{features_path}', 'r') as f:")
    print(f"      features = json.load(f)")
    
    print(f"\nTo view feature importance:")
    print(f"  importance = pd.read_csv('{importance_path}')")
    print(f"  print(importance.head(20))")


# ============================================================================
# EXAMPLE 3: Make predictions on new data
# ============================================================================

def prediction_example():
    """
    Example: Make predictions on new data
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Making Predictions on New Data")
    print("="*70)
    
    print("\nMethod 1: Using the inference script (Recommended)")
    print("-" * 70)
    print("python predict_koi_disposition.py \\")
    print("    --data path/to/new_data.csv \\")
    print("    --model model_outputs/xgboost_koi_disposition_TIMESTAMP.joblib \\")
    print("    --scaler model_outputs/scaler_TIMESTAMP.joblib \\")
    print("    --encoder model_outputs/label_encoder_TIMESTAMP.joblib \\")
    print("    --features model_outputs/feature_names_TIMESTAMP.json \\")
    print("    --output my_predictions.csv")
    
    print("\n\nMethod 2: Programmatically in Python")
    print("-" * 70)
    print("""
import pandas as pd
import joblib
import json

# Load model artifacts
model = joblib.load('model_outputs/xgboost_koi_disposition_TIMESTAMP.joblib')
scaler = joblib.load('model_outputs/scaler_TIMESTAMP.joblib')
encoder = joblib.load('model_outputs/label_encoder_TIMESTAMP.joblib')

with open('model_outputs/feature_names_TIMESTAMP.json', 'r') as f:
    feature_names = json.load(f)

# Load new data
new_data = pd.read_csv('path/to/new_data.csv')

# Select features (same as training)
X_new = new_data[feature_names].fillna(0)

# Scale
X_new_scaled = scaler.transform(X_new)

# Predict
predictions = model.predict(X_new_scaled)
probabilities = model.predict_proba(X_new_scaled)

# Decode predictions
predicted_labels = encoder.inverse_transform(predictions)

# Add to dataframe
new_data['predicted_disposition'] = predicted_labels
new_data['confidence'] = probabilities.max(axis=1)

# Save results
new_data.to_csv('predictions.csv', index=False)
""")


# ============================================================================
# EXAMPLE 4: Analyze model performance
# ============================================================================

def analysis_example():
    """
    Example: Analyze model performance
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Analyzing Model Performance")
    print("="*70)
    
    print("\nTo analyze model performance, check these files:")
    print("\n1. Evaluation Metrics (JSON):")
    print("   model_outputs/evaluation_metrics_TIMESTAMP.json")
    print("   Contains: accuracy, precision, recall, F1-scores, confusion matrix")
    
    print("\n2. Visualization (PNG):")
    print("   model_outputs/model_evaluation_TIMESTAMP.png")
    print("   Contains: confusion matrix, feature importance, training curves, etc.")
    
    print("\n3. Feature Importance (CSV):")
    print("   model_outputs/feature_importance_TIMESTAMP.csv")
    print("   Shows which features contribute most to predictions")
    
    print("\n4. Loading metrics programmatically:")
    print("""
import json

with open('model_outputs/evaluation_metrics_TIMESTAMP.json', 'r') as f:
    metrics = json.load(f)

print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
print(f"CV Mean: {metrics['cv_mean']:.4f} +/- {metrics['cv_std']:.4f}")
print("\\nPer-class metrics:")
for class_name, class_metrics in metrics['classification_report'].items():
    if isinstance(class_metrics, dict):
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall: {class_metrics['recall']:.4f}")
        print(f"    F1-Score: {class_metrics['f1-score']:.4f}")
""")


# ============================================================================
# EXAMPLE 5: Cross-validation and model comparison
# ============================================================================

def cross_validation_example():
    """
    Example: Understanding cross-validation results
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Understanding Cross-Validation")
    print("="*70)
    
    print("\nThe training script automatically performs 5-fold cross-validation.")
    print("This helps ensure the model generalizes well to unseen data.")
    
    print("\nWhat to look for:")
    print("  ✓ CV scores should be close to each other (low variance)")
    print("  ✓ CV mean should be close to test accuracy (no overfitting)")
    print("  ✓ If test accuracy >> CV mean: possible data leakage")
    print("  ✓ If test accuracy << CV mean: possible overfitting")
    
    print("\nExample interpretation:")
    print("  CV scores: [0.89, 0.91, 0.90, 0.88, 0.90]")
    print("  CV mean: 0.896 (+/- 0.011)")
    print("  Test accuracy: 0.893")
    print("  → Good! Model is stable and not overfitted")


# ============================================================================
# EXAMPLE 6: Hyperparameter tuning
# ============================================================================

def hyperparameter_tuning_example():
    """
    Example: How to tune hyperparameters
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Hyperparameter Tuning (Advanced)")
    print("="*70)
    
    print("\nThe default configuration is optimized for the KOI dataset.")
    print("However, you can experiment with different hyperparameters.")
    
    print("\nKey hyperparameters to tune:")
    print("""
1. max_depth (3-10): Controls tree depth
   - Lower: prevents overfitting, might underfit
   - Higher: more complex model, might overfit

2. learning_rate (0.01-0.3): Step size for updates
   - Lower: more conservative, needs more trees
   - Higher: faster learning, might miss optimum

3. n_estimators (100-1000): Number of trees
   - More trees: better performance, longer training
   - Use early_stopping to find optimal

4. subsample (0.5-1.0): Fraction of samples per tree
   - Lower: prevents overfitting, faster training
   - Higher: uses more data per tree

5. colsample_bytree (0.5-1.0): Fraction of features per tree
   - Lower: prevents overfitting, adds randomness
   - Higher: uses more features per tree

6. min_child_weight (1-10): Minimum samples in leaf
   - Lower: more complex trees
   - Higher: more conservative, prevents overfitting

7. gamma (0-1): Minimum loss reduction for split
   - Higher: more conservative splitting
   - Lower: more aggressive splitting

8. reg_alpha (0-1): L1 regularization
   - Higher: more regularization

9. reg_lambda (0-10): L2 regularization
   - Higher: more regularization
""")

    print("\nTo modify hyperparameters, edit train_koi_disposition.py:")
    print("  Look for 'xgb_params' dictionary in the train_model() method")


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Run all examples"""
    print("\n")
    print("*" * 70)
    print(" KOI DISPOSITION PREDICTION - USAGE EXAMPLES")
    print("*" * 70)
    
    train_model_example()
    inspect_model_example()
    prediction_example()
    analysis_example()
    cross_validation_example()
    hyperparameter_tuning_example()
    
    print("\n" + "*" * 70)
    print(" For detailed documentation, see: TRAINING_GUIDE.md")
    print("*" * 70)
    print()


if __name__ == "__main__":
    main()

