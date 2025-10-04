# 🚀 Quick Start Guide - KOI Disposition Prediction

## ⚡ TL;DR - Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_koi_disposition.py
```

### Step 3: Check Results
Look in the `model_outputs/` folder for:
- Trained model (`.joblib` file)
- Evaluation metrics (`.json` file)
- Visualizations (`.png` file)

**That's it!** ✅

---

## 📋 Essential Commands

### Training
```bash
# Train model with default settings (recommended)
python train_koi_disposition.py
```

### Making Predictions
```bash
# Replace TIMESTAMP with the actual timestamp from your trained model files
python predict_koi_disposition.py \
    --data data/koi_with_relative_location.csv \
    --model model_outputs/xgboost_koi_disposition_TIMESTAMP.joblib \
    --scaler model_outputs/scaler_TIMESTAMP.joblib \
    --encoder model_outputs/label_encoder_TIMESTAMP.joblib \
    --features model_outputs/feature_names_TIMESTAMP.json \
    --output predictions.csv
```

### View Examples
```bash
# See all usage examples
python example_usage.py
```

---

## 📊 What to Expect

### Training Output
- **Duration**: 5-15 minutes
- **Files Created**: 6 files in `model_outputs/`
- **Expected Accuracy**: 85-92%

### Key Metrics
- ✅ Test Accuracy
- ✅ Cross-Validation Scores
- ✅ Precision, Recall, F1 per class
- ✅ Confusion Matrix
- ✅ Feature Importance

---

## 🔍 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run: `pip install -r requirements.txt` |
| `FileNotFoundError` | Check that `data/koi_with_relative_location.csv` exists |
| Memory error | Close other applications, or use smaller dataset |
| Low accuracy | Check data quality and class distribution |

---

## 📁 Project Structure

```
.
├── train_koi_disposition.py      # Main training script ⭐
├── predict_koi_disposition.py    # Inference script
├── requirements.txt               # Dependencies
├── TRAINING_GUIDE.md             # Full documentation
├── QUICKSTART.md                 # This file
├── example_usage.py              # Usage examples
├── data/
│   └── koi_with_relative_location.csv  # Dataset
└── model_outputs/                # Created after training
    ├── *.joblib                  # Model files
    ├── *.json                    # Metadata
    ├── *.csv                     # Feature importance
    └── *.png                     # Visualizations
```

---

## 🎯 Next Steps

1. ✅ Train the model (see above)
2. 📊 Review the visualization: `model_outputs/model_evaluation_*.png`
3. 📈 Check metrics: `model_outputs/evaluation_metrics_*.json`
4. 🔮 Make predictions on new data
5. 📚 Read full guide: `TRAINING_GUIDE.md`
6. 🚀 Implement improvements from "Future Work" section

---

## 💡 Pro Tips

- **For Competition**: Optimize the decision threshold for your specific metric
- **Better Performance**: Implement feature engineering from Future Work section
- **Ensemble**: Combine multiple models for better results
- **Interpretability**: Use SHAP values to understand predictions

---

## 📖 Full Documentation

For detailed information, see:
- **TRAINING_GUIDE.md** - Complete guide with best practices
- **example_usage.py** - Code examples
- Comments in `train_koi_disposition.py` - Implementation details

---

## 🆘 Need Help?

1. Check `TRAINING_GUIDE.md` troubleshooting section
2. Review `example_usage.py` for code examples
3. Read inline comments in the training script
4. Check NASA Exoplanet Archive documentation

---

**Good luck with your competition! 🌟🚀**

