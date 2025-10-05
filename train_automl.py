#!/usr/bin/env python3
"""
AutoML Script for KOI Disposition Prediction
============================================
This script automatically tries different machine learning models with various
hyperparameters to find the best performing model for exoplanet classification.

Author: Exoplanet ML Team
Dataset: NASA Exoplanet Archive - KOI Cumulative Table
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

# Import multiple ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier

# GPU and parallel processing imports
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - using basic system info")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - GPU acceleration disabled")

# Try to import GPU-accelerated libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    print("⚠️ CuPy not available - using CPU only")

try:
    import cuml
    CUML_AVAILABLE = True
    print("✅ cuML available for GPU-accelerated ML")
except ImportError:
    CUML_AVAILABLE = False
    print("⚠️ cuML not available - using scikit-learn")

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import os
from itertools import product
import time

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
    'output_dir': 'automl_outputs',
    'n_trials': 50,  # Number of random combinations to try
    'timeout_per_model': 300,  # 5 minutes per model
    
    # GPU and parallel processing configuration
    'use_gpu': True,  # Enable GPU acceleration when available
    'n_jobs': -1,  # Use all available CPU cores (-1 for all cores)
    'parallel_trials': True,  # Run multiple trials in parallel
    'max_parallel_trials': 4,  # Maximum parallel trials (to avoid memory issues)
    'gpu_memory_fraction': 0.8,  # Fraction of GPU memory to use
    'use_mixed_precision': True,  # Use mixed precision for GPU models
}

class AutoMLPipeline:
    """
    Automated Machine Learning pipeline for KOI disposition prediction.
    """
    
    def __init__(self, config):
        self.config = config
        self.results = []
        self.best_model = None
        self.best_score = 0
        self.best_params = None
        
        # GPU and system detection
        self.gpu_available = self._detect_gpu()
        self.cpu_count = mp.cpu_count()
        
        if PSUTIL_AVAILABLE:
            self.available_memory = psutil.virtual_memory().total / (1024**3)  # GB
        else:
            self.available_memory = 8.0  # Default assumption
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Print system information
        self._print_system_info()
    
    def _detect_gpu(self):
        """Detect available GPU resources."""
        gpu_info = {
            'cuda_available': False,
            'cuda_devices': 0,
            'cupy_available': CUPY_AVAILABLE,
            'cuml_available': CUML_AVAILABLE
        }
        
        if TORCH_AVAILABLE:
            gpu_info['cuda_available'] = torch.cuda.is_available()
            gpu_info['cuda_devices'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            
            if gpu_info['cuda_available']:
                gpu_info['cuda_device_name'] = torch.cuda.get_device_name(0)
                gpu_info['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        return gpu_info
    
    def _print_system_info(self):
        """Print system information for optimization."""
        print("\n" + "="*70)
        print("SYSTEM INFORMATION")
        print("="*70)
        
        print(f"🖥️  CPU Cores: {self.cpu_count}")
        print(f"💾 Available Memory: {self.available_memory:.1f} GB")
        
        if self.gpu_available['cuda_available']:
            print(f"🚀 GPU Available: {self.gpu_available['cuda_device_name']}")
            print(f"🎮 GPU Memory: {self.gpu_available['cuda_memory']:.1f} GB")
            print(f"📊 GPU Devices: {self.gpu_available['cuda_devices']}")
        else:
            print("⚠️  No GPU detected - using CPU only")
        
        print(f"🔧 CuPy Available: {'✅' if self.gpu_available['cupy_available'] else '❌'}")
        print(f"🔧 cuML Available: {'✅' if self.gpu_available['cuml_available'] else '❌'}")
        
        # Determine optimal configuration
        if self.gpu_available['cuda_available'] and self.config['use_gpu']:
            print(f"🚀 GPU acceleration enabled")
            self.config['n_jobs'] = min(4, self.cpu_count)  # Reduce CPU jobs when using GPU
        else:
            print(f"🖥️  CPU-only mode - using {self.cpu_count} cores")
            self.config['n_jobs'] = self.cpu_count
        
        print(f"⚡ Parallel trials: {'✅' if self.config['parallel_trials'] else '❌'}")
        print(f"🔢 Max parallel trials: {self.config['max_parallel_trials']}")
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        print("="*70)
        print("AUTOML PIPELINE - DATA LOADING AND PREPROCESSING")
        print("="*70)
        
        # Load data
        df = pd.read_csv(self.config['data_path'])
        print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check target distribution
        print(f"\nTarget distribution:")
        print(df['koi_disposition'].value_counts())
        
        # Preprocessing (simplified version of the original)
        df_processed = df.copy()
        
        # Remove excluded columns (same as original)
        excluded_columns = [
            'koi_delivname', 'koi_vet_stat', 'koi_quarters', 'koi_count', 'koi_max_sngle_ev',
            'koi_max_mult_ev', 'koi_bin_oedp_sig', 'koi_limbdark_mod', 'koi_ldm_coeff1',
            'koi_ldm_coeff2', 'koi_ldm_coeff3', 'koi_ldm_coeff4', 'koi_trans_mod',
            'koi_model_dof', 'koi_model_chisq', 'koi_eccen', 'koi_longp', 'koi_time0',
            'koi_ingress', 'koi_incl', 'koi_sparprov', 'koi_comment', 'koi_vet_date',
            'koi_tce_plnt_num', 'koi_tce_delivname', 'koi_disp_prov', 'koi_parm_prov',
            'host', 'hd_name', 'hip_name', 'glon', 'glat', 'elon', 'elat',
            'sy_icmag', 'sy_icmagerr1', 'sy_icmagerr2', 'sy_bmag', 'sy_vmag', 'sy_umag',
            'sy_pm', 'sy_pmra', 'sy_pmdec', 'use_ra', 'use_dec', 'rowid', 'kepoi_id'
        ]
        
        # Remove excluded columns
        columns_to_remove = [col for col in excluded_columns if col in df_processed.columns]
        df_processed = df_processed.drop(columns=columns_to_remove)
        print(f"✓ Removed {len(columns_to_remove)} excluded columns")
        
        # Separate target variable before removing non-numeric columns
        target_column = 'koi_disposition'
        if target_column in df_processed.columns:
            target = df_processed[target_column]
            df_processed = df_processed.drop(columns=[target_column])
        else:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Remove non-numeric columns
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed = df_processed[numeric_columns]
        print(f"✓ Kept {len(numeric_columns)} numeric columns")
        
        # Remove columns with >50% missing values
        missing_threshold = 0.5
        high_missing = df_processed.columns[df_processed.isnull().mean() > missing_threshold]
        df_processed = df_processed.drop(columns=high_missing)
        print(f"✓ Removed {len(high_missing)} columns with >50% missing values")
        
        # Remove zero variance columns
        zero_var_cols = df_processed.columns[df_processed.var() == 0]
        df_processed = df_processed.drop(columns=zero_var_cols)
        print(f"✓ Removed {len(zero_var_cols)} zero variance columns")
        
        # Remove highly correlated features
        corr_matrix = df_processed.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
        df_processed = df_processed.drop(columns=high_corr_cols)
        print(f"✓ Removed {len(high_corr_cols)} highly correlated features")
        
        print(f"\n✓ Final dataset: {df_processed.shape[0]} rows, {df_processed.shape[1]} features")
        
        return df_processed, target
    
    def get_model_configurations(self):
        """Define model configurations with hyperparameter ranges."""
        # Base parameters for all models
        base_params = {
            'random_state': [RANDOM_STATE],
            'n_jobs': [self.config['n_jobs']]
        }
        
        # GPU-optimized parameters
        gpu_params = {}
        if self.gpu_available['cuda_available'] and self.config['use_gpu']:
            gpu_params = {
                'tree_method': ['gpu_hist'],
                'gpu_id': [0],
                'predictor': ['gpu_predictor']
            }
        
        return {
            'XGBoost': {
                'model': XGBClassifier,
                'params': {
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200, 300, 500],
                    'min_child_weight': [1, 3, 5],
                    'gamma': [0, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [1, 1.5, 2],
                    **base_params,
                    **gpu_params
                }
            },
            'LightGBM': {
                'model': LGBMClassifier,
                'params': {
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [50, 100, 200, 300, 500],
                    'min_child_samples': [10, 20, 30],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 0.5],
                    'device': ['gpu'] if self.gpu_available['cuda_available'] and self.config['use_gpu'] else ['cpu'],
                    'gpu_platform_id': [0] if self.gpu_available['cuda_available'] and self.config['use_gpu'] else [None],
                    'gpu_device_id': [0] if self.gpu_available['cuda_available'] and self.config['use_gpu'] else [None],
                    **base_params,
                    'verbose': [-1]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    'random_state': [RANDOM_STATE]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5, 6],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0],
                    'random_state': [RANDOM_STATE]
                }
            },
            'ExtraTrees': {
                'model': ExtraTreesClassifier,
                'params': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    'random_state': [RANDOM_STATE]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000, 2000],
                    'random_state': [RANDOM_STATE]
                }
            },
            'SVM': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'probability': [True],
                    'random_state': [RANDOM_STATE]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier,
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11, 15],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'NaiveBayes': {
                'model': GaussianNB,
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                }
            },
            'DecisionTree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'criterion': ['gini', 'entropy'],
                    'random_state': [RANDOM_STATE]
                }
            },
            'MLP': {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                    'activation': ['relu', 'tanh', 'logistic'],
                    'solver': ['adam', 'lbfgs', 'sgd'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive'],
                    'max_iter': [500, 1000],
                    'random_state': [RANDOM_STATE]
                }
            }
        }
    
    def generate_random_params(self, param_space, n_samples=1):
        """Generate random parameter combinations."""
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        combinations = []
        for _ in range(n_samples):
            params = {}
            for name, values in param_space.items():
                params[name] = np.random.choice(values)
            combinations.append(params)
        
        return combinations
    
    def _train_single_model(self, trial_info):
        """Train a single model configuration (for parallel processing)."""
        model_class, params, X_train, y_train, X_val, y_val, X_test, y_test, trial_id = trial_info
        
        try:
            print(f"🔬 Trial {trial_id} - {model_class.__name__}")
            
            # Create model instance
            model = model_class(**params)
            
            # Handle class imbalance
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
            
            # Train model
            if hasattr(model, 'sample_weight'):
                sample_weights = np.array([class_weight_dict[int(label)] for label in y_train])
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1)
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            return {
                'trial_id': trial_id,
                'model_name': model_class.__name__,
                'params': params,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error in trial {trial_id}: {str(e)}")
            return {
                'trial_id': trial_id,
                'model_name': model_class.__name__,
                'params': params,
                'error': str(e),
                'success': False
            }
    
    def train_and_evaluate_model(self, model_class, params, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train and evaluate a single model configuration."""
        try:
            # Create model instance
            model = model_class(**params)
            
            # Handle class imbalance
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
            
            # Train model
            if hasattr(model, 'sample_weight'):
                sample_weights = np.array([class_weight_dict[int(label)] for label in y_train])
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            return {
                'model_name': model_class.__name__,
                'params': params,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
        except Exception as e:
            print(f"❌ Error training {model_class.__name__}: {str(e)}")
            return None
    
    def run_automl(self):
        """Run the complete AutoML pipeline."""
        print("\n" + "="*70)
        print("AUTOML PIPELINE - MODEL TRAINING AND EVALUATION")
        print("="*70)
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], 
            stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config['val_size']/(1-self.config['test_size']), 
            random_state=self.config['random_state'], 
            stratify=y_temp
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)
        X_test_imputed = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_val_encoded = label_encoder.transform(y_val)
        y_test_encoded = label_encoder.transform(y_test)
        
        print(f"\n🎯 Target classes: {label_encoder.classes_}")
        
        # Get model configurations
        model_configs = self.get_model_configurations()
        
        print(f"\n🤖 Testing {len(model_configs)} different model types...")
        print(f"📊 Total trials: {self.config['n_trials']}")
        
        # Run AutoML
        start_time = time.time()
        trial_count = 0
        
        # Prepare all trials for parallel processing
        all_trials = []
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*50}")
            print(f"Preparing {model_name} trials")
            print(f"{'='*50}")
            
            model_class = config['model']
            param_space = config['params']
            
            # Generate random parameter combinations
            n_trials_per_model = max(1, self.config['n_trials'] // len(model_configs))
            param_combinations = self.generate_random_params(param_space, n_trials_per_model)
            
            for i, params in enumerate(param_combinations):
                trial_count += 1
                trial_info = (
                    model_class, params, 
                    X_train_scaled, y_train_encoded,
                    X_val_scaled, y_val_encoded,
                    X_test_scaled, y_test_encoded,
                    trial_count
                )
                all_trials.append(trial_info)
        
        print(f"\n🚀 Starting parallel training of {len(all_trials)} trials...")
        
        # Run trials in parallel or sequentially
        if self.config['parallel_trials'] and len(all_trials) > 1:
            max_workers = min(self.config['max_parallel_trials'], len(all_trials), self.cpu_count)
            print(f"⚡ Using {max_workers} parallel workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self._train_single_model, all_trials))
        else:
            print("🔄 Running trials sequentially")
            results = [self._train_single_model(trial) for trial in all_trials]
        
        # Process results
        for result in results:
            if result['success']:
                self.results.append(result)
                
                # Update best model
                if result['cv_mean'] > self.best_score:
                    self.best_score = result['cv_mean']
                    self.best_model = result['model']
                    self.best_params = result['params']
                    print(f"🏆 New best model! CV Score: {result['cv_mean']:.4f}")
                else:
                    print(f"📊 Trial {result['trial_id']} CV Score: {result['cv_mean']:.4f}")
            else:
                print(f"❌ Trial {result['trial_id']} failed: {result.get('error', 'Unknown error')}")
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Total training time: {elapsed_time:.1f} seconds")
        
        print(f"\n{'='*70}")
        print(f"AUTOML COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Tested {len(self.results)} model configurations")
        print(f"✓ Best CV Score: {self.best_score:.4f}")
        print(f"✓ Best Model: {self.best_model.__class__.__name__}")
        
        return self.results, self.best_model, self.best_params
    
    def analyze_results(self):
        """Analyze and visualize AutoML results."""
        if not self.results:
            print("❌ No results to analyze!")
            return
        
        print("\n" + "="*70)
        print("AUTOML RESULTS ANALYSIS")
        print("="*70)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {
                'model_name': r['model_name'],
                'accuracy': r['accuracy'],
                'cv_mean': r['cv_mean'],
                'cv_std': r['cv_std'],
                'f1': r['f1']
            } for r in self.results
        ])
        
        # Sort by CV score
        results_df = results_df.sort_values('cv_mean', ascending=False)
        
        print("\n🏆 TOP 10 MODELS:")
        print("-" * 70)
        top_10 = results_df.head(10)
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            print(f"{i:2d}. {row['model_name']:15s} | CV: {row['cv_mean']:.4f} ± {row['cv_std']:.4f} | "
                  f"Test: {row['accuracy']:.4f} | F1: {row['f1']:.4f}")
        
        # Model type analysis
        print(f"\n📊 MODEL TYPE PERFORMANCE:")
        print("-" * 50)
        model_stats = results_df.groupby('model_name').agg({
            'cv_mean': ['mean', 'max', 'count'],
            'accuracy': 'mean'
        }).round(4)
        
        for model_name in model_stats.index:
            mean_cv = model_stats.loc[model_name, ('cv_mean', 'mean')]
            max_cv = model_stats.loc[model_name, ('cv_mean', 'max')]
            count = model_stats.loc[model_name, ('cv_mean', 'count')]
            mean_test = model_stats.loc[model_name, ('accuracy', 'mean')]
            print(f"{model_name:15s} | Mean CV: {mean_cv:.4f} | Max CV: {max_cv:.4f} | "
                  f"Mean Test: {mean_test:.4f} | Trials: {count}")
        
        # Create visualizations
        self.create_visualizations(results_df)
        
        return results_df
    
    def create_visualizations(self, results_df):
        """Create visualization plots for AutoML results."""
        print("\n📊 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AutoML Results Analysis', fontsize=16, fontweight='bold')
        
        # 1. Model Performance Comparison
        model_performance = results_df.groupby('model_name')['cv_mean'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        
        axes[0, 0].bar(range(len(model_performance)), model_performance['mean'], 
                      yerr=model_performance['std'], capsize=5, alpha=0.7)
        axes[0, 0].set_xticks(range(len(model_performance)))
        axes[0, 0].set_xticklabels(model_performance.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('CV Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. CV Score vs Test Accuracy
        axes[0, 1].scatter(results_df['cv_mean'], results_df['accuracy'], alpha=0.6)
        axes[0, 1].set_xlabel('CV Score')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].set_title('CV Score vs Test Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add diagonal line for perfect correlation
        min_val = min(results_df['cv_mean'].min(), results_df['accuracy'].min())
        max_val = max(results_df['cv_mean'].max(), results_df['accuracy'].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        # 3. Distribution of CV Scores
        axes[1, 0].hist(results_df['cv_mean'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(results_df['cv_mean'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {results_df["cv_mean"].mean():.4f}')
        axes[1, 0].axvline(results_df['cv_mean'].max(), color='green', linestyle='--', 
                          label=f'Max: {results_df["cv_mean"].max():.4f}')
        axes[1, 0].set_xlabel('CV Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of CV Scores')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Top 10 Models
        top_10 = results_df.head(10)
        y_pos = np.arange(len(top_10))
        
        bars = axes[1, 1].barh(y_pos, top_10['cv_mean'], alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f"{row['model_name']}" for _, row in top_10.iterrows()])
        axes[1, 1].set_xlabel('CV Score')
        axes[1, 1].set_title('Top 10 Models')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_10['cv_mean'])):
            axes[1, 1].text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                           f'{value:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.config['output_dir'], f'automl_results_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {plot_path}")
        
        plt.show()
    
    def save_results(self):
        """Save AutoML results to files."""
        if not self.results:
            print("❌ No results to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_data = []
        for result in self.results:
            results_data.append({
                'model_name': result['model_name'],
                'params': result['params'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1': result['f1'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            })
        
        results_path = os.path.join(self.config['output_dir'], f'automl_results_{timestamp}.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"✓ Results saved to: {results_path}")
        
        # Save best model
        if self.best_model:
            model_path = os.path.join(self.config['output_dir'], f'best_model_{timestamp}.joblib')
            joblib.dump(self.best_model, model_path)
            print(f"✓ Best model saved to: {model_path}")
            
            # Save best model info
            best_model_info = {
                'model_name': self.best_model.__class__.__name__,
                'params': self.best_params,
                'cv_score': self.best_score,
                'timestamp': timestamp
            }
            
            info_path = os.path.join(self.config['output_dir'], f'best_model_info_{timestamp}.json')
            with open(info_path, 'w') as f:
                json.dump(best_model_info, f, indent=2)
            print(f"✓ Best model info saved to: {info_path}")


def main():
    """Main execution function."""
    print("🚀 Starting AutoML Pipeline for KOI Disposition Prediction")
    print("="*70)
    
    # Initialize AutoML pipeline
    automl = AutoMLPipeline(CONFIG)
    
    # Run AutoML
    results, best_model, best_params = automl.run_automl()
    
    # Analyze results
    results_df = automl.analyze_results()
    
    # Save results
    automl.save_results()
    
    print(f"\n🎉 AutoML Pipeline Complete!")
    print(f"🏆 Best Model: {best_model.__class__.__name__}")
    print(f"📊 Best CV Score: {automl.best_score:.4f}")
    print(f"📁 Results saved to: {CONFIG['output_dir']}/")
    
    return automl, results_df


if __name__ == "__main__":
    automl, results_df = main()
