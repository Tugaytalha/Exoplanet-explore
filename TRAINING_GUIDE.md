# KOI Disposition Prediction - XGBoost Model Training Guide

## 📋 Overview

This project implements a comprehensive machine learning pipeline for predicting exoplanet dispositions (CONFIRMED, FALSE POSITIVE, CANDIDATE) from NASA's Kepler Objects of Interest (KOI) dataset using XGBoost.

### Dataset
- **Source**: NASA Exoplanet Archive
- **URL**: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
- **Features**: ~295 columns including planet properties, stellar characteristics, and positional data
- **Target**: `koi_disposition` (multiclass classification)

---

## 🚀 Quick Start

### 1. Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

**Required packages:**
- Python 3.8+
- numpy>=1.21.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- xgboost>=1.5.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- joblib>=1.1.0

### 2. Training the Model

Simply run the training script:

```bash
python train_koi_disposition.py
```

**What happens during training:**

1. ✅ **Data Loading & Exploration**: Loads CSV and analyzes target distribution
2. ✅ **Data Preprocessing**: 
   - Removes flags, scores, and identifiers (as requested)
   - Handles missing values intelligently
   - Removes high-correlation features (>0.95)
   - Removes zero-variance features
3. ✅ **Feature Engineering**: 
   - Median imputation for missing values
   - StandardScaler normalization
4. ✅ **Class Imbalance Handling**: Computes balanced class weights
5. ✅ **Model Training**: Trains XGBoost with early stopping
6. ✅ **Cross-Validation**: 5-fold stratified cross-validation
7. ✅ **Evaluation**: Comprehensive metrics on test set
8. ✅ **Visualization**: Generates detailed evaluation plots
9. ✅ **Model Saving**: Saves model and all artifacts

**Training time**: Approximately 5-15 minutes depending on your hardware.

### 3. Output Files

All outputs are saved in the `model_outputs/` directory:

```
model_outputs/
├── xgboost_koi_disposition_TIMESTAMP.joblib    # Trained model
├── scaler_TIMESTAMP.joblib                      # Feature scaler
├── label_encoder_TIMESTAMP.joblib               # Label encoder
├── feature_names_TIMESTAMP.json                 # List of features
├── feature_importance_TIMESTAMP.csv             # Feature importance scores
├── evaluation_metrics_TIMESTAMP.json            # All metrics in JSON
└── model_evaluation_TIMESTAMP.png               # Visualization plots
```

### 4. Making Predictions

Use the inference script to predict on new data:

```bash
python predict_koi_disposition.py \
    --data path/to/new_data.csv \
    --model model_outputs/xgboost_koi_disposition_TIMESTAMP.joblib \
    --scaler model_outputs/scaler_TIMESTAMP.joblib \
    --encoder model_outputs/label_encoder_TIMESTAMP.joblib \
    --features model_outputs/feature_names_TIMESTAMP.json \
    --output predictions_output.csv
```

The output CSV will include:
- Original data columns
- `predicted_disposition`: Predicted class
- `prob_CANDIDATE`: Probability of CANDIDATE class
- `prob_CONFIRMED`: Probability of CONFIRMED class
- `prob_FALSE POSITIVE`: Probability of FALSE POSITIVE class
- `prediction_confidence`: Confidence score (max probability)

---

## 🎯 Model Architecture & Methodology

### XGBoost Configuration

```python
{
    'objective': 'multi:softprob',          # Multi-class classification
    'max_depth': 6,                         # Tree depth (prevents overfitting)
    'learning_rate': 0.05,                  # Conservative learning rate
    'n_estimators': 500,                    # With early stopping
    'min_child_weight': 3,                  # Regularization
    'gamma': 0.1,                           # Minimum loss reduction
    'subsample': 0.8,                       # Sample 80% of data
    'colsample_bytree': 0.8,                # Sample 80% of features
    'reg_alpha': 0.1,                       # L1 regularization
    'reg_lambda': 1.0,                      # L2 regularization
    'early_stopping_rounds': 50             # Stop if no improvement
}
```

### Best Practices Implemented

#### 1. **Data Preprocessing**
- ✅ Removes data leakage features (disposition scores, flags)
- ✅ Handles missing values with median imputation
- ✅ Removes highly correlated features (correlation > 0.95)
- ✅ Removes zero-variance features
- ✅ Standardizes features using StandardScaler

#### 2. **Feature Engineering**
- ✅ Automatic feature selection based on variance and correlation
- ✅ Preserves all informative features
- ✅ Handles both continuous and categorical variables

#### 3. **Model Training**
- ✅ Stratified train/validation/test split (70%/10%/20%)
- ✅ Class imbalance handling with sample weights
- ✅ Early stopping to prevent overfitting
- ✅ Cross-validation for robust evaluation

#### 4. **Evaluation**
- ✅ Multiple metrics: Accuracy, Precision, Recall, F1-Score
- ✅ Confusion matrix analysis
- ✅ Per-class performance metrics
- ✅ Cross-validation scores
- ✅ Feature importance analysis

#### 5. **Visualization**
- ✅ Confusion matrix heatmap
- ✅ Top features importance plot
- ✅ Training/validation loss curves
- ✅ Cross-validation scores
- ✅ Class distribution analysis
- ✅ Per-class performance metrics

---

## 📊 Expected Performance

Based on the KOI dataset characteristics:

- **Overall Accuracy**: ~85-92%
- **CONFIRMED Class**: High precision (>90%)
- **FALSE POSITIVE Class**: Good precision (>85%)
- **CANDIDATE Class**: Moderate precision (varies based on data quality)

**Note**: Performance varies based on:
- Dataset version and quality
- Class distribution
- Feature availability
- Random seed initialization

---

## 🔍 Feature Importance

The model automatically identifies the most important features. Typically, key features include:

1. **Transit Properties**:
   - `koi_depth`: Transit depth
   - `koi_period`: Orbital period
   - `koi_duration`: Transit duration
   - `koi_prad`: Planet radius

2. **Stellar Properties**:
   - `koi_steff`: Stellar effective temperature
   - `koi_srad`: Stellar radius
   - `koi_slogg`: Stellar surface gravity

3. **Signal Quality**:
   - `koi_model_snr`: Signal-to-noise ratio
   - `koi_num_transits`: Number of transits observed
   - `koi_max_mult_ev`: Maximum multiple event statistic

---

## 🐛 Troubleshooting

### Issue: Missing columns error
**Solution**: Ensure your CSV has the same structure as the original KOI dataset. The script handles missing features gracefully.

### Issue: Memory error during training
**Solution**: Reduce dataset size or use a machine with more RAM. The KOI dataset is ~118MB and should work on most systems.

### Issue: Poor model performance
**Solutions**:
1. Check class distribution in your data
2. Ensure data quality (remove corrupted rows)
3. Try different hyperparameters
4. Collect more training data

### Issue: ImportError
**Solution**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

---

## 📈 Model Interpretability (Optional)

To understand model predictions better, you can use SHAP (SHapley Additive exPlanations):

```python
import shap

# Load your trained model
model = joblib.load('model_outputs/xgboost_koi_disposition_TIMESTAMP.joblib')

# Create explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_scaled)

# Visualize
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names)
```

---

## 🔬 Future Work

### 1. **Advanced Feature Engineering** (High Priority)
- [ ] Create domain-specific features:
  - Planet-to-star radius ratio transformations
  - Orbital mechanics-derived features
  - Temperature equilibrium calculations
  - Stellar luminosity estimations
- [ ] Polynomial features for key parameters
- [ ] Interaction terms between physical properties
- [ ] Time-series features from light curve data (if available)

### 2. **Hyperparameter Optimization** (High Priority)
- [ ] Implement Bayesian Optimization (using Optuna or Hyperopt)
- [ ] Grid search over extended parameter space
- [ ] Automated hyperparameter tuning pipeline
- [ ] Learning rate scheduling
- [ ] Tree pruning optimization

### 3. **Ensemble Methods** (High Priority)
- [ ] Stack XGBoost with other models:
  - Random Forest
  - LightGBM
  - CatBoost
  - Neural Networks
- [ ] Implement voting classifiers
- [ ] Create model blending pipeline
- [ ] Use model uncertainty for ensemble selection

### 4. **Class Imbalance Handling** (Medium Priority)
- [ ] Implement SMOTE (Synthetic Minority Over-sampling)
- [ ] Use ADASYN (Adaptive Synthetic Sampling)
- [ ] Try focal loss for hard examples
- [ ] Experiment with cost-sensitive learning
- [ ] Threshold optimization for each class

### 5. **Deep Learning Approaches** (Medium Priority)
- [ ] Implement neural networks:
  - Multi-layer Perceptron (MLP)
  - 1D Convolutional Neural Networks
  - Attention mechanisms
- [ ] Use raw light curve data directly (if available)
- [ ] Transfer learning from pre-trained models
- [ ] Ensemble deep learning with XGBoost

### 6. **Model Interpretability** (Medium Priority)
- [ ] Implement SHAP values analysis
- [ ] LIME (Local Interpretable Model-agnostic Explanations)
- [ ] Partial Dependence Plots (PDP)
- [ ] Individual Conditional Expectation (ICE) plots
- [ ] Feature interaction detection

### 7. **Data Quality Improvements** (Medium Priority)
- [ ] Outlier detection and removal (Isolation Forest)
- [ ] Advanced missing value imputation (KNN, iterative)
- [ ] Feature transformation (Box-Cox, Yeo-Johnson)
- [ ] Robust scaling for outliers
- [ ] Data augmentation techniques

### 8. **Validation Strategy** (Medium Priority)
- [ ] Implement time-based validation (if temporal aspect matters)
- [ ] Nested cross-validation for hyperparameter selection
- [ ] Leave-one-group-out validation (by stellar host)
- [ ] Bootstrap confidence intervals
- [ ] Calibration analysis (Platt scaling, isotonic regression)

### 9. **Production Deployment** (Low Priority)
- [ ] Create REST API (Flask/FastAPI)
- [ ] Dockerize the application
- [ ] Implement model versioning (MLflow)
- [ ] Create monitoring dashboard
- [ ] Set up automated retraining pipeline
- [ ] A/B testing framework

### 10. **Documentation & Reporting** (Low Priority)
- [ ] Create detailed technical report
- [ ] Generate automated model cards
- [ ] Build interactive web dashboard (Streamlit/Dash)
- [ ] Create Jupyter notebooks for exploration
- [ ] Write scientific paper/blog post

### 11. **Advanced Techniques** (Research)
- [ ] Multi-task learning (predict multiple properties)
- [ ] Semi-supervised learning using CANDIDATE labels
- [ ] Active learning for data labeling
- [ ] Uncertainty quantification
- [ ] Conformal prediction for confidence intervals
- [ ] Domain adaptation techniques

### 12. **Competition-Specific Optimizations**
- [ ] Analyze competition metric carefully
- [ ] Optimize decision threshold per class
- [ ] Create submission format pipeline
- [ ] Implement cross-validation strategies used by winners
- [ ] Study domain-specific papers on exoplanet detection

---

## 📚 References & Resources

### Academic Papers
1. **Kepler Mission**: https://www.nasa.gov/mission_pages/kepler/overview/index.html
2. **XGBoost Paper**: Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"
3. **ExoMiner**: Hamed Valizadegan et al. (2021) - "ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier..."

### Useful Links
- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
- Kepler Data Documentation: https://archive.stsci.edu/kepler/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Scikit-learn Best Practices: https://scikit-learn.org/stable/

### Similar Projects
- ExoMiner (NASA): Deep learning for exoplanet classification
- Kaggle Exoplanet Hunting: Various competition solutions
- TESS Data Analysis: Modern space telescope data

---

## 🤝 Contributing

To improve this project:

1. Implement items from the Future Work section
2. Experiment with different model architectures
3. Add unit tests and integration tests
4. Improve documentation
5. Share your results and insights

---

## 📝 License

This project uses NASA public datasets which are in the public domain. The code is provided for educational and research purposes.

---

## 💡 Tips for Competition Success

1. **Understand the Metric**: Know exactly what metric your competition uses (accuracy, F1, AUC-ROC, etc.)
2. **Feature Engineering is Key**: Domain knowledge of exoplanets helps create powerful features
3. **Ensemble Multiple Models**: Combine predictions from different models
4. **Cross-Validation**: Use robust CV strategy to avoid overfitting
5. **Handle Class Imbalance**: Use appropriate techniques for imbalanced classes
6. **Optimize Thresholds**: Don't use default 0.5 threshold, optimize per class
7. **Use All Data**: Combine train and validation for final submission
8. **Study Winners' Solutions**: After competition, learn from top performers

---

## 🎓 Learning Resources

If you're new to XGBoost or exoplanet science:

1. **XGBoost Tutorial**: https://xgboost.readthedocs.io/en/stable/tutorials/model.html
2. **Exoplanet Detection Methods**: https://exoplanets.nasa.gov/alien-worlds/ways-to-find-a-planet/
3. **Machine Learning for Astronomy**: Coursera "Data-driven Astronomy" course
4. **Transit Method**: Understanding how Kepler detected planets

---

**Good luck with your competition! 🚀🌟**

For questions or issues, please refer to the troubleshooting section or review the code comments.

