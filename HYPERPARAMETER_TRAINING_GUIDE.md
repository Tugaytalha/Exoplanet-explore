# 🎯 Hyperparameter Training Guide

## Overview

The `/api/train` endpoint now supports **custom hyperparameters** for XGBoost model training. This allows you to fine-tune the model for better performance on your specific dataset.

---

## 🚀 Quick Start

### Basic Training (Default Hyperparameters)

```bash
curl -X POST "http://localhost:8000/api/train" \
  -F "training_file=@my_data.csv" \
  -F "model_name=my_model"
```

### Training with Custom Hyperparameters

```bash
curl -X POST "http://localhost:8000/api/train" \
  -F "training_file=@my_data.csv" \
  -F "model_name=tuned_model" \
  -F "max_depth=8" \
  -F "learning_rate=0.1" \
  -F "n_estimators=300" \
  -F "subsample=0.9" \
  -F "colsample_bytree=0.9"
```

---

## 📊 Available Hyperparameters

### XGBoost Model Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_depth` | 6 | 3-10 | Maximum depth of trees. Higher = more complex |
| `learning_rate` | 0.05 | 0.01-0.3 | Step size shrinkage. Lower = more robust |
| `n_estimators` | 500 | 100-1000 | Number of boosting rounds |
| `min_child_weight` | 3 | 1-10 | Minimum sum of instance weight in child |
| `gamma` | 0.1 | 0-1 | Minimum loss reduction for split |
| `subsample` | 0.8 | 0.5-1.0 | Fraction of samples for training |
| `colsample_bytree` | 0.8 | 0.5-1.0 | Fraction of features for training |
| `reg_alpha` | 0.1 | 0-1 | L1 regularization term |
| `reg_lambda` | 1.0 | 0-10 | L2 regularization term |

### Training Configuration

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `test_size` | 0.2 | 0.1-0.3 | Test set proportion |
| `val_size` | 0.1 | 0.1-0.2 | Validation set proportion |
| `n_folds` | 5 | 3-10 | Cross-validation folds |
| `early_stopping_rounds` | 50 | 20-100 | Rounds before early stopping |

---

## 🎓 Hyperparameter Tuning Guide

### Understanding the Parameters

#### 1. **max_depth** (Tree Complexity)
Controls how deep each tree can grow.

**Higher values (8-10):**
- ✅ Can capture complex patterns
- ✅ Better for large datasets
- ❌ Risk of overfitting
- ❌ Slower training

**Lower values (3-5):**
- ✅ Faster training
- ✅ Less overfitting
- ❌ May underfit complex data

**Recommended:**
- Start with 6
- Increase if underfitting (both train/test accuracy low)
- Decrease if overfitting (train high, test low)

#### 2. **learning_rate** (Learning Speed)
Controls how much each tree contributes.

**Higher values (0.1-0.3):**
- ✅ Faster convergence
- ✅ Fewer trees needed
- ❌ May overshoot optimal solution

**Lower values (0.01-0.05):**
- ✅ More stable learning
- ✅ Better final performance
- ❌ Needs more trees
- ❌ Slower training

**Recommended:**
- Start with 0.05
- Use 0.01-0.03 with more `n_estimators` for best accuracy
- Use 0.1-0.3 for faster experimentation

#### 3. **n_estimators** (Number of Trees)
Number of boosting rounds.

**More trees (800-1000):**
- ✅ Better performance (with low learning rate)
- ❌ Longer training time
- ❌ Risk of overfitting (without regularization)

**Fewer trees (100-300):**
- ✅ Faster training
- ❌ May underfit

**Recommended:**
- Start with 500
- Increase with lower learning rate
- Use early stopping to find optimal number

#### 4. **subsample & colsample_bytree** (Sampling)
Control randomness in training.

**Higher values (0.9-1.0):**
- ✅ Use more data
- ❌ Less regularization

**Lower values (0.5-0.7):**
- ✅ More regularization
- ✅ Prevents overfitting
- ❌ May lose information

**Recommended:**
- Start with 0.8 for both
- Decrease if overfitting
- Try different combinations

#### 5. **reg_alpha & reg_lambda** (Regularization)
Control model complexity penalty.

**Higher regularization:**
- ✅ Prevents overfitting
- ✅ More generalizable
- ❌ May underfit

**Lower regularization:**
- ✅ More flexible model
- ❌ Risk of overfitting

**Recommended:**
- Start with `reg_alpha=0.1`, `reg_lambda=1.0`
- Increase if overfitting
- Decrease if underfitting

---

## 🔍 Troubleshooting Performance Issues

### Scenario 1: Underfitting (Low Accuracy Overall)

**Symptoms:**
- Train accuracy: ~75%
- Test accuracy: ~74%
- Both low!

**Solutions:**
```bash
# Try these changes:
-F "max_depth=8"           # Was: 6 → Increase complexity
-F "n_estimators=800"      # Was: 500 → More rounds
-F "min_child_weight=1"    # Was: 3 → Less restrictive
-F "reg_alpha=0.05"        # Was: 0.1 → Less regularization
-F "reg_lambda=0.5"        # Was: 1.0 → Less regularization
```

### Scenario 2: Overfitting (High Train, Low Test)

**Symptoms:**
- Train accuracy: ~99%
- Test accuracy: ~85%
- Big gap!

**Solutions:**
```bash
# Try these changes:
-F "max_depth=4"           # Was: 6 → Reduce complexity
-F "learning_rate=0.03"    # Was: 0.05 → Slower, more stable
-F "subsample=0.7"         # Was: 0.8 → More randomness
-F "colsample_bytree=0.7"  # Was: 0.8 → Feature sampling
-F "reg_alpha=0.3"         # Was: 0.1 → More regularization
-F "reg_lambda=2.0"        # Was: 1.0 → More regularization
```

### Scenario 3: Slow Training

**Symptoms:**
- Takes too long to train
- Need faster iterations

**Solutions:**
```bash
# Try these changes:
-F "n_estimators=200"      # Was: 500 → Fewer trees
-F "learning_rate=0.1"     # Was: 0.05 → Faster convergence
-F "n_folds=3"             # Was: 5 → Faster CV
-F "early_stopping_rounds=30"  # Was: 50 → Stop earlier
```

### Scenario 4: Want Best Accuracy (Time Not Critical)

**Symptoms:**
- Have time and compute
- Want maximum performance

**Solutions:**
```bash
# Try these changes:
-F "n_estimators=1000"     # More trees
-F "learning_rate=0.01"    # Slow and steady
-F "max_depth=7"           # Moderate complexity
-F "subsample=0.85"        # Use more data
-F "colsample_bytree=0.85" # Use more features
-F "n_folds=10"            # More thorough CV
```

---

## 📝 Example Use Cases

### Example 1: Quick Baseline

```bash
curl -X POST "http://localhost:8000/api/train" \
  -F "training_file=@data.csv" \
  -F "model_name=baseline" \
  -F "n_estimators=200" \
  -F "learning_rate=0.1" \
  -F "n_folds=3"
```

### Example 2: Production Model (High Accuracy)

```bash
curl -X POST "http://localhost:8000/api/train" \
  -F "training_file=@data.csv" \
  -F "model_name=production" \
  -F "max_depth=7" \
  -F "learning_rate=0.02" \
  -F "n_estimators=1000" \
  -F "subsample=0.85" \
  -F "colsample_bytree=0.85" \
  -F "reg_alpha=0.1" \
  -F "reg_lambda=1.5" \
  -F "n_folds=10" \
  -F "early_stopping_rounds=75"
```

### Example 3: Regularized Model (Prevent Overfitting)

```bash
curl -X POST "http://localhost:8000/api/train" \
  -F "training_file=@data.csv" \
  -F "model_name=regularized" \
  -F "max_depth=5" \
  -F "learning_rate=0.03" \
  -F "n_estimators=600" \
  -F "subsample=0.7" \
  -F "colsample_bytree=0.7" \
  -F "reg_alpha=0.3" \
  -F "reg_lambda=2.0"
```

### Example 4: Complex Model (Large Dataset)

```bash
curl -X POST "http://localhost:8000/api/train" \
  -F "training_file=@large_data.csv" \
  -F "model_name=complex" \
  -F "max_depth=9" \
  -F "learning_rate=0.05" \
  -F "n_estimators=800" \
  -F "min_child_weight=2" \
  -F "gamma=0.05" \
  -F "subsample=0.9" \
  -F "colsample_bytree=0.9"
```

---

## 🐍 Python Example

```python
import requests

# Define hyperparameters
hyperparameters = {
    'training_file': open('my_data.csv', 'rb'),
    'model_name': 'tuned_model',
    # Model complexity
    'max_depth': 7,
    'min_child_weight': 2,
    # Learning
    'learning_rate': 0.03,
    'n_estimators': 800,
    # Regularization
    'gamma': 0.1,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.15,
    'reg_lambda': 1.5,
    # Training config
    'test_size': 0.2,
    'val_size': 0.1,
    'n_folds': 7,
    'early_stopping_rounds': 60
}

# Train model
response = requests.post(
    'http://localhost:8000/api/train',
    files={'training_file': hyperparameters.pop('training_file')},
    data=hyperparameters
)

result = response.json()

# Check results
print(f"✅ Training complete!")
print(f"Test Accuracy: {result['metrics']['test_accuracy']:.4f}")
print(f"CV Mean: {result['metrics']['cv_mean']:.4f} ± {result['metrics']['cv_std']:.4f}")
print(f"\nHyperparameters used:")
for key, value in result['hyperparameters'].items():
    print(f"  {key}: {value}")
```

---

## 📊 Response Structure

The training endpoint returns:

```json
{
  "status": "success",
  "model_name": "tuned_model",
  "timestamp": "20251005_123456",
  "hyperparameters": {
    "max_depth": 7,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "min_child_weight": 2,
    "gamma": 0.1,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.15,
    "reg_lambda": 1.5,
    "early_stopping_rounds": 60,
    "test_size": 0.2,
    "val_size": 0.1,
    "n_folds": 7
  },
  "metrics": {
    "test_accuracy": 0.9845,
    "test_precision": 0.9852,
    "test_recall": 0.9845,
    "test_f1": 0.9848,
    "cv_mean": 0.9851,
    "cv_std": 0.0008,
    "cv_scores": [0.9843, 0.9850, 0.9859, 0.9847, 0.9855]
  },
  "classification_report": {...},
  "confusion_matrix": [...],
  "feature_importance": [...],
  "model_files": {...},
  "visualization": {...}
}
```

---

## 🎯 Best Practices

### 1. Start Simple
Begin with default parameters, then tune based on results.

### 2. Tune One at a Time
Change one parameter, observe results, then adjust next.

### 3. Use Cross-Validation
Higher `n_folds` gives more reliable estimates (but slower).

### 4. Monitor Train vs. Test
- Similar scores = good generalization
- Large gap = overfitting
- Both low = underfitting

### 5. Balance Speed vs. Accuracy
- Development: Fast iterations (fewer estimators, higher LR)
- Production: Best accuracy (more estimators, lower LR)

### 6. Save Results
Keep track of what hyperparameters work best for your data.

---

## 🔬 Advanced: Grid Search

For systematic tuning, try multiple combinations:

```python
import requests
import itertools

# Define parameter grid
param_grid = {
    'max_depth': [5, 6, 7],
    'learning_rate': [0.03, 0.05, 0.1],
    'n_estimators': [500, 800]
}

results = []

# Try all combinations
for max_depth, learning_rate, n_estimators in itertools.product(
    param_grid['max_depth'],
    param_grid['learning_rate'],
    param_grid['n_estimators']
):
    print(f"Testing: max_depth={max_depth}, lr={learning_rate}, n_est={n_estimators}")
    
    response = requests.post(
        'http://localhost:8000/api/train',
        files={'training_file': open('data.csv', 'rb')},
        data={
            'model_name': f'grid_d{max_depth}_lr{learning_rate}_n{n_estimators}',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators
        }
    )
    
    result = response.json()
    results.append({
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'test_accuracy': result['metrics']['test_accuracy'],
        'cv_mean': result['metrics']['cv_mean']
    })

# Find best parameters
best = max(results, key=lambda x: x['cv_mean'])
print(f"\n🏆 Best parameters:")
print(f"   max_depth: {best['max_depth']}")
print(f"   learning_rate: {best['learning_rate']}")
print(f"   n_estimators: {best['n_estimators']}")
print(f"   CV Mean: {best['cv_mean']:.4f}")
```

---

## 📚 Additional Resources

- [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [Hyperparameter Tuning Guide](https://www.kaggle.com/prashant111/a-guide-on-xgboost-hyperparameters-tuning)
- [Understanding XGBoost](https://towardsdatascience.com/understanding-xgboost-hyperparameters-e2a0e5d7e26d)

---

## 🎉 Summary

You now have full control over XGBoost training! Experiment with different hyperparameters to find the best configuration for your data. Remember:

1. **Start with defaults**
2. **Identify the problem** (underfitting/overfitting/speed)
3. **Adjust relevant parameters**
4. **Validate with cross-validation**
5. **Iterate until satisfied**

Happy tuning! 🚀

