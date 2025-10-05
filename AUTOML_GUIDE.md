# 🤖 AutoML Guide for KOI Disposition Prediction

## Overview

The `train_automl.py` script automatically tests multiple machine learning models with various hyperparameters to find the best performing model for exoplanet classification.

## Features

### 🎯 **11 Different ML Models**
- **XGBoost** - Gradient boosting with advanced features
- **LightGBM** - Fast gradient boosting
- **Random Forest** - Ensemble of decision trees
- **Gradient Boosting** - Traditional gradient boosting
- **Extra Trees** - Extremely randomized trees
- **Logistic Regression** - Linear classification
- **SVM** - Support Vector Machine
- **K-Nearest Neighbors** - Instance-based learning
- **Naive Bayes** - Probabilistic classifier
- **Decision Tree** - Single tree classifier
- **MLP** - Multi-layer perceptron (neural network)

### 🔧 **Hyperparameter Optimization**
- **Random Search** - Tests random combinations of hyperparameters
- **Configurable Trials** - Set number of trials per model
- **Timeout Protection** - Prevents infinite training
- **Class Imbalance Handling** - Automatic sample weighting

### 📊 **Comprehensive Evaluation**
- **Cross-Validation** - 5-fold CV for robust evaluation
- **Multiple Metrics** - Accuracy, Precision, Recall, F1-score
- **Model Comparison** - Side-by-side performance analysis
- **Visualization** - Automatic plotting of results

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the Pipeline
```bash
python test_automl.py
```

### 3. Run Full AutoML
```bash
python train_automl.py
```

## Configuration

### Basic Configuration
```python
CONFIG = {
    'data_path': 'data/koi_with_relative_location.csv',
    'test_size': 0.2,           # 20% for testing
    'val_size': 0.1,            # 10% for validation
    'n_folds': 5,               # Cross-validation folds
    'n_trials': 50,             # Total trials across all models
    'timeout_per_model': 300,   # 5 minutes per model type
    'output_dir': 'automl_outputs'
}
```

### Advanced Configuration
```python
# Customize for your needs
CONFIG.update({
    'n_trials': 100,           # More trials for better results
    'timeout_per_model': 600,  # 10 minutes per model
    'n_folds': 10,             # More CV folds
    'test_size': 0.15,         # Smaller test set
    'val_size': 0.15           # Larger validation set
})
```

## Model Hyperparameters

### XGBoost Parameters
```python
'max_depth': [3, 4, 5, 6, 7, 8]
'learning_rate': [0.01, 0.05, 0.1, 0.2]
'n_estimators': [50, 100, 200, 300, 500]
'min_child_weight': [1, 3, 5]
'gamma': [0, 0.1, 0.2]
'subsample': [0.8, 0.9, 1.0]
'colsample_bytree': [0.8, 0.9, 1.0]
'reg_alpha': [0, 0.1, 0.5]
'reg_lambda': [1, 1.5, 2]
```

### LightGBM Parameters
```python
'max_depth': [3, 4, 5, 6, 7, 8]
'learning_rate': [0.01, 0.05, 0.1, 0.2]
'n_estimators': [50, 100, 200, 300, 500]
'min_child_samples': [10, 20, 30]
'subsample': [0.8, 0.9, 1.0]
'colsample_bytree': [0.8, 0.9, 1.0]
'reg_alpha': [0, 0.1, 0.5]
'reg_lambda': [0, 0.1, 0.5]
```

### Random Forest Parameters
```python
'n_estimators': [50, 100, 200, 300, 500]
'max_depth': [3, 5, 7, 10, None]
'min_samples_split': [2, 5, 10]
'min_samples_leaf': [1, 2, 4]
'max_features': ['sqrt', 'log2', None]
'bootstrap': [True, False]
```

## Output Files

### 📁 **Generated Files**
```
automl_outputs/
├── automl_results_YYYYMMDD_HHMMSS.json    # Detailed results
├── best_model_YYYYMMDD_HHMMSS.joblib      # Best trained model
├── best_model_info_YYYYMMDD_HHMMSS.json   # Best model metadata
└── automl_results_YYYYMMDD_HHMMSS.png     # Visualization plots
```

### 📊 **Results JSON Structure**
```json
{
  "model_name": "XGBClassifier",
  "params": {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200
  },
  "accuracy": 0.9994,
  "precision": 0.9993,
  "recall": 0.9994,
  "f1": 0.9993,
  "cv_mean": 0.9992,
  "cv_std": 0.0003
}
```

## Visualization Plots

### 📈 **4 Analysis Plots**
1. **Model Performance Comparison** - Bar chart of average CV scores
2. **CV Score vs Test Accuracy** - Scatter plot showing correlation
3. **Distribution of CV Scores** - Histogram of all scores
4. **Top 10 Models** - Horizontal bar chart of best performers

## Usage Examples

### Basic Usage
```python
from train_automl import AutoMLPipeline, CONFIG

# Use default configuration
automl = AutoMLPipeline(CONFIG)
results, best_model, best_params = automl.run_automl()
```

### Custom Configuration
```python
# Custom configuration
custom_config = CONFIG.copy()
custom_config.update({
    'n_trials': 20,
    'timeout_per_model': 120,
    'output_dir': 'my_automl_results'
})

automl = AutoMLPipeline(custom_config)
results, best_model, best_params = automl.run_automl()
```

### Analyze Results
```python
# Analyze and visualize results
results_df = automl.analyze_results()

# Save results
automl.save_results()

# Access best model
print(f"Best model: {best_model.__class__.__name__}")
print(f"Best CV score: {automl.best_score:.4f}")
```

## Performance Tips

### 🚀 **Speed Optimization**
1. **Reduce Trials**: Lower `n_trials` for faster results
2. **Timeout Settings**: Set reasonable `timeout_per_model`
3. **Parallel Processing**: Models use `n_jobs=-1` when available
4. **Memory Management**: Large datasets may need chunking

### 🎯 **Accuracy Optimization**
1. **More Trials**: Increase `n_trials` for better exploration
2. **More CV Folds**: Higher `n_folds` for robust evaluation
3. **Longer Timeouts**: Allow complex models to train fully
4. **Feature Engineering**: Add domain-specific features

## Troubleshooting

### Common Issues

#### 1. **Memory Errors**
```python
# Reduce dataset size for testing
CONFIG['test_size'] = 0.3  # Larger test set = smaller training set
```

#### 2. **Timeout Issues**
```python
# Increase timeout for complex models
CONFIG['timeout_per_model'] = 600  # 10 minutes
```

#### 3. **Import Errors**
```bash
# Install missing dependencies
pip install lightgbm
pip install scikit-learn>=1.0.0
```

#### 4. **Data Issues**
```python
# Check data file exists
import os
if not os.path.exists('data/koi_with_relative_location.csv'):
    print("❌ Data file not found. Run fetch.py first.")
```

## Advanced Usage

### Custom Model Addition
```python
def get_model_configurations(self):
    configs = super().get_model_configurations()
    
    # Add custom model
    configs['MyCustomModel'] = {
        'model': MyCustomClassifier,
        'params': {
            'param1': [1, 2, 3],
            'param2': [0.1, 0.5, 1.0]
        }
    }
    
    return configs
```

### Custom Metrics
```python
def train_and_evaluate_model(self, model_class, params, X_train, y_train, X_val, y_val, X_test, y_test):
    # ... existing code ...
    
    # Add custom metrics
    custom_metric = self.calculate_custom_metric(y_test, y_pred)
    
    return {
        # ... existing metrics ...
        'custom_metric': custom_metric
    }
```

## Results Interpretation

### 🏆 **Best Model Selection**
- **Primary Metric**: Cross-validation mean score
- **Secondary Metrics**: Test accuracy, F1-score
- **Stability**: Low CV standard deviation preferred

### 📊 **Model Comparison**
- **Tree-based models** (XGBoost, LightGBM, Random Forest) typically perform best
- **Linear models** (Logistic Regression) are fast but may underperform
- **Neural networks** (MLP) can be powerful but need more data

### 🎯 **Hyperparameter Insights**
- **High learning rates** → Faster training, potential overfitting
- **Deep trees** → More complex models, potential overfitting
- **Regularization** → Prevents overfitting, improves generalization

## Integration with API

### Use Best Model in API
```python
# Load best model from AutoML results
import joblib
best_model = joblib.load('automl_outputs/best_model_YYYYMMDD_HHMMSS.joblib')

# Use in API prediction
prediction = best_model.predict(features)
```

### Model Comparison
```python
# Compare AutoML results with manual tuning
automl_score = 0.9994
manual_score = 0.9987

if automl_score > manual_score:
    print("🏆 AutoML found a better model!")
else:
    print("📊 Manual tuning performed better")
```

## Next Steps

1. **Run AutoML**: `python train_automl.py`
2. **Analyze Results**: Review generated plots and JSON files
3. **Select Best Model**: Use the highest CV score model
4. **Integrate with API**: Load best model for production use
5. **Monitor Performance**: Track model performance over time

---

**Happy AutoML-ing! 🚀**
