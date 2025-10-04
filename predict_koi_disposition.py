"""
Inference Script for KOI Disposition Prediction
================================================
This script loads a trained XGBoost model and makes predictions on new data.

Usage:
    python predict_koi_disposition.py --data path/to/new_data.csv --model model_outputs/xgboost_koi_disposition_TIMESTAMP.joblib
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import os
from datetime import datetime


class KOIDispositionInference:
    """
    Inference pipeline for predicting KOI dispositions.
    """
    
    def __init__(self, model_path, scaler_path, encoder_path, features_path):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
            encoder_path: Path to the label encoder
            features_path: Path to the feature names JSON
        """
        print("Loading model artifacts...")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(encoder_path)
        
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"✓ Model loaded successfully")
        print(f"  - Features: {len(self.feature_names)}")
        print(f"  - Classes: {self.label_encoder.classes_}")
    
    def preprocess_data(self, df):
        """
        Preprocess input data to match training format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed feature matrix
        """
        print("\nPreprocessing data...")
        
        # Select only the features used in training
        available_features = [f for f in self.feature_names if f in df.columns]
        missing_features = [f for f in self.feature_names if f not in df.columns]
        
        if missing_features:
            print(f"⚠ Warning: {len(missing_features)} features missing from input data")
            print(f"  Missing features will be filled with zeros")
        
        # Create feature matrix
        X = pd.DataFrame(columns=self.feature_names)
        
        # Fill available features
        for feat in available_features:
            X[feat] = df[feat]
        
        # Fill missing features with 0
        for feat in missing_features:
            X[feat] = 0
        
        # Handle missing values (impute with median from training would be better,
        # but for simplicity we use 0 here)
        X = X.fillna(0)
        
        # Convert to numpy array
        X = X.values
        
        print(f"✓ Preprocessing complete: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X
    
    def predict(self, X):
        """
        Make predictions on preprocessed data.
        
        Args:
            X: Preprocessed feature matrix
            
        Returns:
            predictions: Predicted class labels
            probabilities: Prediction probabilities for each class
        """
        print("\nMaking predictions...")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred_encoded = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)
        
        # Decode labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        print(f"✓ Predictions complete")
        
        return y_pred, y_pred_proba
    
    def predict_from_file(self, data_path, output_path=None):
        """
        Load data from file and make predictions.
        
        Args:
            data_path: Path to input CSV file
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        print(f"\nLoading data from: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Preprocess
        X = self.preprocess_data(df)
        
        # Predict
        predictions, probabilities = self.predict(X)
        
        # Create results DataFrame
        results = df.copy()
        results['predicted_disposition'] = predictions
        
        # Add probability columns
        for i, class_name in enumerate(self.label_encoder.classes_):
            results[f'prob_{class_name}'] = probabilities[:, i]
        
        # Add prediction confidence (max probability)
        results['prediction_confidence'] = probabilities.max(axis=1)
        
        # Display prediction summary
        print("\n" + "="*70)
        print("PREDICTION SUMMARY")
        print("="*70)
        print(f"\nPredicted class distribution:")
        print(results['predicted_disposition'].value_counts())
        print(f"\nMean prediction confidence: {results['prediction_confidence'].mean():.4f}")
        
        # Save results
        if output_path:
            results.to_csv(output_path, index=False)
            print(f"\n✓ Predictions saved to: {output_path}")
        
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Predict KOI dispositions using trained XGBoost model'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.joblib)'
    )
    
    parser.add_argument(
        '--scaler',
        type=str,
        required=True,
        help='Path to fitted scaler file (.joblib)'
    )
    
    parser.add_argument(
        '--encoder',
        type=str,
        required=True,
        help='Path to label encoder file (.joblib)'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to feature names file (.json)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"predictions_{timestamp}.csv"
    
    # Initialize inference pipeline
    inference = KOIDispositionInference(
        model_path=args.model,
        scaler_path=args.scaler,
        encoder_path=args.encoder,
        features_path=args.features
    )
    
    # Make predictions
    results = inference.predict_from_file(args.data, args.output)
    
    print("\n✓ Inference complete!")
    
    return results


if __name__ == "__main__":
    results = main()

