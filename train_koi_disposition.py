"""
XGBoost Model Training for KOI Disposition Prediction
======================================================
This script trains an XGBoost classifier to predict exoplanet disposition
from NASA's Kepler Objects of Interest (KOI) dataset.

Author: Exoplanet ML Team
Dataset: NASA Exoplanet Archive - KOI Cumulative Table
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import os

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configuration
CONFIG = {
    'data_path': 'data/koi_with_relative_location.csv',
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': RANDOM_STATE,
    'n_folds': 5,
    'output_dir': 'model_outputs',
    'model_name': 'xgboost_koi_disposition',
}


class KOIDispositionPredictor:
    """
    A comprehensive machine learning pipeline for predicting KOI dispositions.
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.imputer = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.feature_importance = None
        self.training_history = {}
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
    def load_and_explore_data(self):
        """Load data and perform initial exploration."""
        print("="*70)
        print("STEP 1: LOADING AND EXPLORING DATA")
        print("="*70)
        
        df = pd.read_csv(self.config['data_path'])
        print(f"\n✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check target distribution
        print(f"\nTarget variable distribution:")
        print(df['koi_disposition'].value_counts())
        print(f"\nPercentage distribution:")
        print(df['koi_disposition'].value_counts(normalize=True) * 100)
        
        return df
    
    def preprocess_data(self, df):
        """
        Comprehensive data preprocessing pipeline.
        - Remove excluded columns (flags, scores, identifiers)
        - Handle missing values
        - Feature selection
        - Encode categorical variables
        """
        print("\n" + "="*70)
        print("STEP 2: DATA PREPROCESSING")
        print("="*70)
        
        # Make a copy
        df_processed = df.copy()
        
        # 1. Identify and remove excluded columns
        print("\n1. Removing excluded columns...")
        
        # Columns to exclude based on requirements and best practices
        exclude_patterns = [
            'fpflag',  # False positive flags (as requested)
            'score',   # Disposition scores (as requested)
            'kepid', 'kepoi_name', 'kepler_name',  # Identifiers
            'ra_str', 'dec_str', 'rastr', 'decstr',  # String representations
            'koi_comment',  # Text comments
            'koi_vet_date',  # Dates
            'koi_delivname', 'koi_tce_delivname', 'koi_disp_prov', 'koi_parm_prov',  # Metadata
            'koi_datalink',  # URLs
            'host', 'hostname', 'hd_name', 'hip_name', 'tic_id',  # Alternative names
            'st_refname', 'sy_refname',  # References
            'sy_name',  # System name
            'gaia_id',  # Alternative IDs
            'koi_pdisposition',  # Related to disposition (data leakage)
            'koi_vet_stat',  # Vetting status (data leakage potential)
        ]
        
        columns_to_drop = []
        for col in df_processed.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in exclude_patterns):
                columns_to_drop.append(col)
        
        # Also drop the target from features
        if 'koi_disposition' in df_processed.columns:
            y = df_processed['koi_disposition'].copy()
            columns_to_drop.append('koi_disposition')
        
        print(f"   - Dropping {len(columns_to_drop)} columns")
        df_features = df_processed.drop(columns=columns_to_drop, errors='ignore')
        
        # 2. Handle non-numeric columns
        print("\n2. Handling non-numeric columns...")
        
        # Identify non-numeric columns
        non_numeric_cols = df_features.select_dtypes(include=['object']).columns.tolist()
        
        if non_numeric_cols:
            print(f"   - Found {len(non_numeric_cols)} non-numeric columns")
            print(f"   - Columns: {non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}")
            
            # Try to convert to numeric, drop if not possible
            for col in non_numeric_cols:
                # Try converting to numeric
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
            
            # Drop columns that are still object type or all NaN
            remaining_non_numeric = df_features.select_dtypes(include=['object']).columns.tolist()
            if remaining_non_numeric:
                print(f"   - Dropping {len(remaining_non_numeric)} columns that couldn't be converted")
                df_features = df_features.drop(columns=remaining_non_numeric)
        
        # 3. Remove columns with too many missing values
        print("\n3. Handling missing values...")
        
        missing_threshold = 0.5  # Drop columns with >50% missing
        missing_percentages = df_features.isnull().sum() / len(df_features)
        high_missing_cols = missing_percentages[missing_percentages > missing_threshold].index.tolist()
        
        if high_missing_cols:
            print(f"   - Dropping {len(high_missing_cols)} columns with >{missing_threshold*100}% missing values")
            df_features = df_features.drop(columns=high_missing_cols)
        
        # 4. Remove columns with zero variance
        print("\n4. Removing zero-variance features...")
        
        variance = df_features.var()
        zero_var_cols = variance[variance == 0].index.tolist()
        
        if zero_var_cols:
            print(f"   - Dropping {len(zero_var_cols)} zero-variance columns")
            df_features = df_features.drop(columns=zero_var_cols)
        
        # 5. Remove highly correlated features
        print("\n5. Removing highly correlated features...")
        
        # Fill NaN temporarily for correlation calculation
        df_temp = df_features.fillna(df_features.median())
        correlation_matrix = df_temp.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than 0.95
        high_corr_threshold = 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > high_corr_threshold)]
        
        if to_drop:
            print(f"   - Dropping {len(to_drop)} highly correlated features (r > {high_corr_threshold})")
            df_features = df_features.drop(columns=to_drop)
        
        print(f"\n✓ Preprocessing complete: {df_features.shape[1]} features retained")
        
        # Store feature names
        self.feature_names = df_features.columns.tolist()
        
        return df_features, y
    
    def handle_class_imbalance(self, y_train_encoded, label_encoder):
        """Calculate class weights to handle imbalanced classes."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train_encoded)
        weights = compute_class_weight('balanced', classes=classes, y=y_train_encoded)
        # Create dictionary with integer keys
        class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        
        # Print with class names for clarity
        print(f"\nClass imbalance handling:")
        for encoded_label, weight in class_weights.items():
            class_name = label_encoder.classes_[encoded_label]
            print(f"   {class_name}: {weight:.4f}")
        
        return class_weights
    
    def prepare_features(self, X_train, X_val, X_test):
        """
        Impute missing values and scale features.
        """
        print("\n" + "="*70)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*70)
        
        print("\n1. Imputing missing values...")
        # Use median imputation for numeric features
        self.imputer = SimpleImputer(strategy='median')
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_val_imputed = self.imputer.transform(X_val)
        X_test_imputed = self.imputer.transform(X_test)
        
        # Count total missing values before imputation
        if isinstance(X_train, pd.DataFrame):
            missing_count = X_train.isnull().sum().sum()
        else:
            missing_count = np.isnan(X_train).sum()
        print(f"   - Imputed {missing_count} missing values in training set")
        
        print("\n2. Scaling features...")
        # Standardization
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_val_scaled = self.scaler.transform(X_val_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        print(f"   - Features scaled using StandardScaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model with hyperparameter tuning.
        """
        print("\n" + "="*70)
        print("STEP 4: MODEL TRAINING")
        print("="*70)
        
        # Encode labels
        print("\n1. Encoding target labels...")
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        print(f"   - Classes: {self.label_encoder.classes_}")
        
        # Calculate class weights (after encoding)
        class_weights = self.handle_class_imbalance(y_train_encoded, self.label_encoder)
        
        # Calculate sample weights
        sample_weights = np.array([class_weights[int(label)] for label in y_train_encoded])
        
        # XGBoost parameters - optimized for exoplanet classification
        print("\n2. Training XGBoost model...")
        
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': len(self.label_encoder.classes_),
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_weight': 3,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 50,
        }
        
        print(f"\n   Model parameters:")
        for key, value in xgb_params.items():
            if key not in ['n_jobs', 'random_state']:
                print(f"   - {key}: {value}")
        
        # Train model with validation set for early stopping
        self.model = XGBClassifier(**xgb_params)
        
        eval_set = [(X_train, y_train_encoded), (X_val, y_val_encoded)]
        
        self.model.fit(
            X_train, 
            y_train_encoded,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False
        )
        
        # Get best iteration
        best_iteration = self.model.best_iteration
        print(f"\n✓ Training complete!")
        print(f"   - Best iteration: {best_iteration}")
        print(f"   - Training samples: {len(X_train)}")
        
        # Store training history
        self.training_history = {
            'train_logloss': self.model.evals_result()['validation_0']['mlogloss'],
            'val_logloss': self.model.evals_result()['validation_1']['mlogloss']
        }
        
        return y_val_encoded
    
    def cross_validate_model(self, X, y):
        """Perform cross-validation to assess model robustness."""
        print("\n" + "="*70)
        print("STEP 5: CROSS-VALIDATION")
        print("="*70)
        
        # Encode labels
        y_encoded = self.label_encoder.transform(y)
        
        # Calculate class weights for the combined data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_encoded)
        weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
        class_weights = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        
        # Stratified K-Fold
        skf = StratifiedKFold(
            n_splits=self.config['n_folds'],
            shuffle=True,
            random_state=RANDOM_STATE
        )
        
        cv_scores = []
        fold_num = 1
        
        print(f"\nPerforming {self.config['n_folds']}-fold cross-validation...")
        
        for train_idx, val_idx in skf.split(X, y_encoded):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y_encoded[train_idx], y_encoded[val_idx]
            
            # Calculate sample weights
            sample_weights = np.array([class_weights[int(label)] for label in y_fold_train])
            
            # Train model
            fold_model = XGBClassifier(
                objective='multi:softprob',
                num_class=len(self.label_encoder.classes_),
                max_depth=6,
                learning_rate=0.05,
                n_estimators=300,
                min_child_weight=3,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            
            fold_model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights, verbose=False)
            
            # Evaluate
            score = fold_model.score(X_fold_val, y_fold_val)
            cv_scores.append(score)
            print(f"   Fold {fold_num}: Accuracy = {score:.4f}")
            fold_num += 1
        
        print(f"\n✓ Cross-validation complete!")
        print(f"   - Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        
        return cv_scores
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation."""
        print("\n" + "="*70)
        print("STEP 6: MODEL EVALUATION")
        print("="*70)
        
        # Encode test labels
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Accuracy
        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"\n✓ Test Set Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        print(classification_report(
            y_test_encoded, 
            y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n" + "-"*70)
        print("TOP 20 MOST IMPORTANT FEATURES")
        print("-"*70)
        print(self.feature_importance.head(20).to_string(index=False))
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_test_encoded, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True
            ),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def visualize_results(self, results, cv_scores):
        """Create comprehensive visualizations."""
        print("\n" + "="*70)
        print("STEP 7: GENERATING VISUALIZATIONS")
        print("="*70)
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(2, 3, 1)
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            ax=ax1
        )
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # 2. Feature Importance (Top 15)
        ax2 = plt.subplot(2, 3, 2)
        top_features = self.feature_importance.head(15)
        ax2.barh(range(len(top_features)), top_features['importance'])
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'], fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel('Importance', fontsize=12)
        ax2.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Training History (Loss curves)
        if self.training_history:
            ax3 = plt.subplot(2, 3, 3)
            epochs = range(len(self.training_history['train_logloss']))
            ax3.plot(epochs, self.training_history['train_logloss'], label='Train Loss', linewidth=2)
            ax3.plot(epochs, self.training_history['val_logloss'], label='Val Loss', linewidth=2)
            ax3.set_xlabel('Iteration', fontsize=12)
            ax3.set_ylabel('Log Loss', fontsize=12)
            ax3.set_title('Training History', fontsize=14, fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
        
        # 4. Cross-Validation Scores
        ax4 = plt.subplot(2, 3, 4)
        folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
        colors = ['#2ecc71' if score > np.mean(cv_scores) else '#e74c3c' for score in cv_scores]
        ax4.bar(folds, cv_scores, color=colors, alpha=0.7)
        ax4.axhline(np.mean(cv_scores), color='blue', linestyle='--', linewidth=2, label='Mean')
        ax4.set_ylabel('Accuracy', fontsize=12)
        ax4.set_title('Cross-Validation Scores', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim([min(cv_scores) - 0.02, max(cv_scores) + 0.02])
        
        # 5. Class Distribution in Predictions
        ax5 = plt.subplot(2, 3, 5)
        pred_labels = self.label_encoder.inverse_transform(results['predictions'])
        pred_counts = pd.Series(pred_labels).value_counts()
        colors_pie = plt.cm.Set3(range(len(pred_counts)))
        ax5.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%',
                startangle=90, colors=colors_pie)
        ax5.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
        
        # 6. Performance Metrics by Class
        ax6 = plt.subplot(2, 3, 6)
        class_report = results['classification_report']
        classes = [c for c in class_report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics = ['precision', 'recall', 'f1-score']
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [class_report[c][metric] for c in classes]
            ax6.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax6.set_ylabel('Score', fontsize=12)
        ax6.set_title('Performance Metrics by Class', fontsize=14, fontweight='bold')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels(classes, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        ax6.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.config['output_dir'], f'model_evaluation_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualizations saved to: {plot_path}")
        
        plt.close()
    
    def save_model_artifacts(self, results, cv_scores):
        """Save model and related artifacts."""
        print("\n" + "="*70)
        print("STEP 8: SAVING MODEL ARTIFACTS")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(
            self.config['output_dir'],
            f"{self.config['model_name']}_{timestamp}.joblib"
        )
        joblib.dump(self.model, model_path)
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save imputer
        imputer_path = os.path.join(
            self.config['output_dir'],
            f"imputer_{timestamp}.joblib"
        )
        joblib.dump(self.imputer, imputer_path)
        print(f"✓ Imputer saved to: {imputer_path}")
        
        # Save scaler
        scaler_path = os.path.join(
            self.config['output_dir'],
            f"scaler_{timestamp}.joblib"
        )
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to: {scaler_path}")
        
        # Save label encoder
        encoder_path = os.path.join(
            self.config['output_dir'],
            f"label_encoder_{timestamp}.joblib"
        )
        joblib.dump(self.label_encoder, encoder_path)
        print(f"✓ Label encoder saved to: {encoder_path}")
        
        # Save feature names
        features_path = os.path.join(
            self.config['output_dir'],
            f"feature_names_{timestamp}.json"
        )
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"✓ Feature names saved to: {features_path}")
        
        # Save feature importance
        importance_path = os.path.join(
            self.config['output_dir'],
            f"feature_importance_{timestamp}.csv"
        )
        self.feature_importance.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved to: {importance_path}")
        
        # Save evaluation metrics
        metrics = {
            'timestamp': timestamp,
            'test_accuracy': float(results['accuracy']),
            'cv_scores': [float(score) for score in cv_scores],
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'classification_report': results['classification_report'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'config': self.config
        }
        
        metrics_path = os.path.join(
            self.config['output_dir'],
            f"evaluation_metrics_{timestamp}.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Evaluation metrics saved to: {metrics_path}")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'encoder_path': encoder_path,
            'features_path': features_path,
            'importance_path': importance_path,
            'metrics_path': metrics_path
        }
    
    def run_full_pipeline(self):
        """Execute the complete training pipeline."""
        print("\n")
        print("="*70)
        print(" KOI DISPOSITION PREDICTION - XGBOOST MODEL TRAINING")
        print("="*70)
        print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load data
        df = self.load_and_explore_data()
        
        # Step 2: Preprocess
        X, y = self.preprocess_data(df)
        
        # Split data
        print("\n" + "="*70)
        print("DATA SPLITTING")
        print("="*70)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=RANDOM_STATE,
            stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = self.config['val_size'] / (1 - self.config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
        
        print(f"\n✓ Data split complete:")
        print(f"   - Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   - Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   - Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Step 3: Feature engineering
        X_train_scaled, X_val_scaled, X_test_scaled = self.prepare_features(
            X_train, X_val, X_test
        )
        
        # Step 4: Train model (includes class imbalance handling)
        y_val_encoded = self.train_model(
            X_train_scaled, y_train,
            X_val_scaled, y_val
        )
        
        # Step 5: Cross-validation
        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = pd.concat([y_train, y_val])
        cv_scores = self.cross_validate_model(X_combined, y_combined)
        
        # Step 6: Evaluation
        results = self.evaluate_model(X_test_scaled, y_test)
        
        # Step 7: Visualizations
        self.visualize_results(results, cv_scores)
        
        # Step 8: Save artifacts
        saved_paths = self.save_model_artifacts(results, cv_scores)
        
        # Final summary
        print("\n" + "="*70)
        print(" TRAINING COMPLETE - SUMMARY")
        print("="*70)
        print(f"\n✓ Model successfully trained and saved!")
        print(f"\nKey Metrics:")
        print(f"   - Test Accuracy: {results['accuracy']:.4f}")
        print(f"   - CV Mean Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"   - Number of features: {len(self.feature_names)}")
        print(f"   - Number of classes: {len(self.label_encoder.classes_)}")
        
        print(f"\nAll artifacts saved to: {self.config['output_dir']}/")
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        return results, saved_paths


def main():
    """Main execution function."""
    # Initialize predictor
    predictor = KOIDispositionPredictor(CONFIG)
    
    # Run full pipeline
    results, saved_paths = predictor.run_full_pipeline()
    
    return predictor, results, saved_paths


if __name__ == "__main__":
    predictor, results, saved_paths = main()

