# Latest Training Visualization Endpoint

## 📊 GET `/api/latest-visualization`

Returns the latest training visualization plot as base64-encoded PNG along with complete evaluation metrics.

---

## 📥 Response Structure

```json
{
  "plot_base64": "iVBORw0KGgoAAAANSUhEUgAABQAAAALQCAYAAADPfd...",
  "timestamp": "20251004_221246",
  "filename": "model_evaluation_20251004_221245.png",
  "created_at": "2025-10-04T22:12:45",
  "file_size_bytes": 245678,
  "image_format": "png",
  "description": "Training visualization with confusion matrix, feature importance, class distribution, and CV scores",
  
  "metrics_timestamp": "20251004_221246",
  "metrics": {
    "test_accuracy": 0.9816250842886042,
    "test_precision": 0.9829077369609194,
    "test_recall": 0.9816250842886042,
    "test_f1": 0.982078157117884,
    "cv_mean": 0.9843631329952762,
    "cv_std": 0.0010078224396790842,
    "cv_scores": [
      0.9843009166578864,
      0.9850384574860394,
      0.9858798735511064,
      0.9834562697576397,
      0.9831401475237092
    ]
  },
  
  "classification_report": {
    "CANDIDATE": {
      "precision": 0.6978260869565217,
      "recall": 0.8106060606060606,
      "f1-score": 0.75,
      "support": 396.0
    },
    "CONFIRMED": {
      "precision": 1.0,
      "recall": 0.9984761904761905,
      "f1-score": 0.9992375142966069,
      "support": 10500.0
    },
    "FALSE POSITIVE": {
      "precision": 0.9141304347826087,
      "recall": 0.868801652892562,
      "f1-score": 0.8908898305084746,
      "support": 968.0
    },
    "accuracy": 0.9816250842886042,
    "macro avg": {
      "precision": 0.8706521739130434,
      "recall": 0.8926279679916043,
      "f1-score": 0.8800424482683605,
      "support": 11864.0
    },
    "weighted avg": {
      "precision": 0.9829077369609194,
      "recall": 0.9816250842886042,
      "f1-score": 0.982078157117884,
      "support": 11864.0
    }
  },
  
  "confusion_matrix": [
    [321, 0, 75],
    [12, 10484, 4],
    [127, 0, 841]
  ],
  
  "feature_importance": [
    {"feature": "sy_pmdec", "importance": 0.21147513},
    {"feature": "glat", "importance": 0.092841916},
    {"feature": "glon", "importance": 0.08794094},
    {"feature": "elon", "importance": 0.08669526},
    {"feature": "elat", "importance": 0.056359436},
    {"feature": "sy_jmag", "importance": 0.042989165},
    {"feature": "sy_gaiamagerr1", "importance": 0.02419272},
    {"feature": "sy_tmagerr1", "importance": 0.02351656},
    {"feature": "sy_vmag", "importance": 0.020887418},
    {"feature": "sy_pm", "importance": 0.019456856},
    {"feature": "koi_prad", "importance": 0.019175664},
    {"feature": "sy_pnum", "importance": 0.017356},
    {"feature": "sy_pmerr1", "importance": 0.017152872},
    {"feature": "koi_model_snr", "importance": 0.017060706},
    {"feature": "sy_dist", "importance": 0.015694173},
    {"feature": "koi_dikco_msky", "importance": 0.014408065},
    {"feature": "sy_w4mag", "importance": 0.012072752},
    {"feature": "sy_w2magerr1", "importance": 0.009377256},
    {"feature": "sy_hmagerr1", "importance": 0.008857732},
    {"feature": "x_pc", "importance": 0.0087258285}
  ],
  "feature_importance_count": 115
}
```

---

## 🔍 Response Fields

### Plot Data
- **`plot_base64`** (string): Base64-encoded PNG image
- **`timestamp`** (string): Timestamp from filename (format: `YYYYMMDD_HHMMSS`)
- **`filename`** (string): Original PNG filename
- **`created_at`** (string): ISO 8601 timestamp of file creation
- **`file_size_bytes`** (integer): Size of the PNG file in bytes
- **`image_format`** (string): Image format (always `"png"`)
- **`description`** (string): Description of the visualization contents

### Evaluation Metrics
- **`metrics_timestamp`** (string): Timestamp of the metrics file
- **`metrics`** (object):
  - **`test_accuracy`** (float): Overall accuracy on test set (0-1)
  - **`test_precision`** (float): Weighted average precision (0-1)
  - **`test_recall`** (float): Weighted average recall (0-1)
  - **`test_f1`** (float): Weighted average F1-score (0-1)
  - **`cv_mean`** (float): Mean cross-validation accuracy (0-1)
  - **`cv_std`** (float): Standard deviation of CV scores
  - **`cv_scores`** (array): Individual fold accuracies from 5-fold CV

### Classification Report
- **`classification_report`** (object): Per-class and aggregate metrics
  - Per-class: `CANDIDATE`, `CONFIRMED`, `FALSE POSITIVE`
  - Aggregates: `accuracy`, `macro avg`, `weighted avg`
  - Each contains: `precision`, `recall`, `f1-score`, `support`

### Confusion Matrix
- **`confusion_matrix`** (array): 3×3 matrix showing predictions vs actual
  - Rows: Actual classes (CANDIDATE, CONFIRMED, FALSE POSITIVE)
  - Columns: Predicted classes (CANDIDATE, CONFIRMED, FALSE POSITIVE)
  - Example: `[[321, 0, 75], [12, 10484, 4], [127, 0, 841]]`

### Feature Importance
- **`feature_importance`** (array): Top 20 most important features
  - Each entry: `{"feature": "name", "importance": 0.123}`
  - Sorted by importance (descending)
- **`feature_importance_count`** (integer): Total number of features used

---

## 💻 Usage Examples

### cURL
```bash
# Get complete response
curl "http://localhost:8000/api/latest-visualization"

# Extract metrics only
curl "http://localhost:8000/api/latest-visualization" | jq '.metrics'

# Extract feature importance
curl "http://localhost:8000/api/latest-visualization" | jq '.feature_importance'

# Save plot to file
curl "http://localhost:8000/api/latest-visualization" | \
  jq -r '.plot_base64' | base64 -d > training_plot.png

# Get test accuracy
curl -s "http://localhost:8000/api/latest-visualization" | \
  jq '.metrics.test_accuracy'
```

### Python
```python
import requests
import base64
import json
from PIL import Image
import io

# Get visualization data
response = requests.get('http://localhost:8000/api/latest-visualization')
data = response.json()

# Print metrics summary
print(f"📊 Training Results ({data['timestamp']})")
print(f"   Test Accuracy:  {data['metrics']['test_accuracy']:.4f}")
print(f"   Test Precision: {data['metrics']['test_precision']:.4f}")
print(f"   Test Recall:    {data['metrics']['test_recall']:.4f}")
print(f"   Test F1-Score:  {data['metrics']['test_f1']:.4f}")
print(f"   CV Mean:        {data['metrics']['cv_mean']:.4f} ± {data['metrics']['cv_std']:.4f}")

# Print top 5 features
print("\n🔝 Top 5 Most Important Features:")
for i, feat in enumerate(data['feature_importance'][:5], 1):
    print(f"   {i}. {feat['feature']:20s} {feat['importance']:.4f}")

# Print confusion matrix
print("\n📈 Confusion Matrix:")
cm = data['confusion_matrix']
classes = ['CANDIDATE', 'CONFIRMED', 'FALSE POS']
print(f"             {'   '.join(f'{c:10s}' for c in classes)}")
for i, row in enumerate(cm):
    print(f"{classes[i]:10s} {' '.join(f'{v:10d}' for v in row)}")

# Decode and save plot
image_data = base64.b64decode(data['plot_base64'])
with open('training_plot.png', 'wb') as f:
    f.write(image_data)
print(f"\n💾 Plot saved to training_plot.png")

# Or display in notebook
image = Image.open(io.BytesIO(image_data))
# image.show()  # Opens in default viewer
```

### JavaScript / React
```javascript
// Fetch visualization data
fetch('http://localhost:8000/api/latest-visualization')
  .then(res => res.json())
  .then(data => {
    console.log('Test Accuracy:', data.metrics.test_accuracy);
    console.log('CV Mean:', data.metrics.cv_mean);
    
    // Display plot
    const img = document.getElementById('training-plot');
    img.src = `data:image/png;base64,${data.plot_base64}`;
    
    // Show metrics
    document.getElementById('accuracy').textContent = 
      (data.metrics.test_accuracy * 100).toFixed(2) + '%';
    document.getElementById('cv-score').textContent = 
      (data.metrics.cv_mean * 100).toFixed(2) + '% ± ' + 
      (data.metrics.cv_std * 100).toFixed(2) + '%';
    
    // Build feature importance table
    const tbody = document.getElementById('features-tbody');
    data.feature_importance.forEach(feat => {
      const row = tbody.insertRow();
      row.insertCell(0).textContent = feat.feature;
      row.insertCell(1).textContent = (feat.importance * 100).toFixed(2) + '%';
    });
  });
```

### React Component
```jsx
import { useState, useEffect } from 'react';

function TrainingDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetch('http://localhost:8000/api/latest-visualization')
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error loading visualization:', err);
        setLoading(false);
      });
  }, []);
  
  if (loading) return <div>Loading training results...</div>;
  if (!data) return <div>Failed to load training results</div>;
  
  const { metrics, classification_report, feature_importance } = data;
  
  return (
    <div className="training-dashboard">
      <h1>Training Results - {data.timestamp}</h1>
      
      {/* Metrics Summary */}
      <div className="metrics-grid">
        <MetricCard 
          title="Test Accuracy" 
          value={(metrics.test_accuracy * 100).toFixed(2) + '%'} 
        />
        <MetricCard 
          title="Precision" 
          value={(metrics.test_precision * 100).toFixed(2) + '%'} 
        />
        <MetricCard 
          title="Recall" 
          value={(metrics.test_recall * 100).toFixed(2) + '%'} 
        />
        <MetricCard 
          title="F1-Score" 
          value={(metrics.test_f1 * 100).toFixed(2) + '%'} 
        />
      </div>
      
      {/* Cross-Validation */}
      <div className="cv-section">
        <h2>Cross-Validation</h2>
        <p>
          Mean: {(metrics.cv_mean * 100).toFixed(2)}% ± 
          {(metrics.cv_std * 100).toFixed(2)}%
        </p>
        <div className="cv-scores">
          {metrics.cv_scores.map((score, i) => (
            <span key={i}>Fold {i+1}: {(score * 100).toFixed(2)}%</span>
          ))}
        </div>
      </div>
      
      {/* Visualization Plot */}
      <div className="plot-section">
        <h2>Training Visualization</h2>
        <img 
          src={`data:image/png;base64,${data.plot_base64}`}
          alt="Training Visualization"
          style={{ width: '100%', maxWidth: '1200px' }}
        />
      </div>
      
      {/* Feature Importance */}
      <div className="features-section">
        <h2>Top Features</h2>
        <table>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Importance</th>
            </tr>
          </thead>
          <tbody>
            {feature_importance.map((feat, i) => (
              <tr key={i}>
                <td>{feat.feature}</td>
                <td>{(feat.importance * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Classification Report */}
      <div className="classification-section">
        <h2>Per-Class Performance</h2>
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(classification_report)
              .filter(([key]) => !key.includes('avg') && key !== 'accuracy')
              .map(([cls, metrics]) => (
                <tr key={cls}>
                  <td>{cls}</td>
                  <td>{(metrics.precision * 100).toFixed(2)}%</td>
                  <td>{(metrics.recall * 100).toFixed(2)}%</td>
                  <td>{(metrics['f1-score'] * 100).toFixed(2)}%</td>
                  <td>{metrics.support}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MetricCard({ title, value }) {
  return (
    <div className="metric-card">
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}
```

---

## 🎨 Visualization Contents

The returned PNG contains a 2×2 grid of subplots:

### 1. Confusion Matrix (Top-Left)
- Heatmap showing actual vs predicted classes
- Annotated with counts in each cell
- Color intensity indicates frequency

### 2. Feature Importance (Top-Right)
- Horizontal bar chart of top 20 features
- Sorted by importance (descending)
- Shows which features most influence predictions

### 3. Class Distribution (Bottom-Left)
- Bar chart showing training set balance
- Displays count for each class: CANDIDATE, CONFIRMED, FALSE POSITIVE
- Helps understand class imbalance

### 4. Cross-Validation Scores (Bottom-Right)
- Bar chart of accuracy across 5 folds
- Shows model consistency
- Horizontal line indicates mean accuracy

---

## 🔄 Workflow Integration

### 1. Training Pipeline
```python
# Train model
response = requests.post('http://localhost:8000/api/train', 
                        files={'file': open('training_data.csv', 'rb')})
train_results = response.json()

# Get visualization
viz_response = requests.get('http://localhost:8000/api/latest-visualization')
viz_data = viz_response.json()

# Compare metrics
print(f"Training accuracy: {train_results['accuracy']:.4f}")
print(f"Test accuracy:     {viz_data['metrics']['test_accuracy']:.4f}")
```

### 2. Model Monitoring Dashboard
```python
# Periodically check latest training results
import time

while True:
    try:
        response = requests.get('http://localhost:8000/api/latest-visualization')
        data = response.json()
        
        # Log metrics
        log_metrics({
            'timestamp': data['created_at'],
            'accuracy': data['metrics']['test_accuracy'],
            'cv_mean': data['metrics']['cv_mean'],
            'cv_std': data['metrics']['cv_std']
        })
        
        # Alert if performance drops
        if data['metrics']['test_accuracy'] < 0.95:
            send_alert(f"Model accuracy dropped to {data['metrics']['test_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error checking metrics: {e}")
    
    time.sleep(3600)  # Check every hour
```

### 3. Automated Reporting
```python
# Generate daily report with latest training results
def generate_training_report():
    response = requests.get('http://localhost:8000/api/latest-visualization')
    data = response.json()
    
    # Create HTML report
    html = f"""
    <html>
    <head><title>Training Report - {data['timestamp']}</title></head>
    <body>
        <h1>Daily Training Report</h1>
        <h2>Summary Metrics</h2>
        <ul>
            <li>Test Accuracy: {data['metrics']['test_accuracy']:.4f}</li>
            <li>CV Mean: {data['metrics']['cv_mean']:.4f} ± {data['metrics']['cv_std']:.4f}</li>
            <li>Total Features: {data['feature_importance_count']}</li>
        </ul>
        
        <h2>Visualization</h2>
        <img src="data:image/png;base64,{data['plot_base64']}" width="100%">
        
        <h2>Top 10 Features</h2>
        <ol>
            {''.join(f"<li>{f['feature']}: {f['importance']:.4f}</li>" 
                     for f in data['feature_importance'][:10])}
        </ol>
    </body>
    </html>
    """
    
    # Save or email report
    with open(f"training_report_{data['timestamp']}.html", 'w') as f:
        f.write(html)
    
    return html
```

---

## ⚠️ Error Handling

### No Visualizations Found (404)
```json
{
  "detail": "No training visualizations found. Train a model first."
}
```
**Solution**: Train a model using `POST /api/train` or run `train_koi_disposition.py`

### Metrics File Missing
If metrics JSON is not found, only plot data is returned (no `metrics` field).

### Feature Importance Missing
If feature importance CSV is not found, no `feature_importance` field is included.

---

## 📊 Understanding the Metrics

### Test Metrics
- **Accuracy**: Overall correct predictions / total predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Cross-Validation
- **cv_mean**: Average accuracy across all folds
- **cv_std**: Consistency of model performance
- **cv_scores**: Individual fold results (5 folds)

### Good Performance Indicators
- ✅ Test accuracy > 0.95 (95%)
- ✅ CV std < 0.01 (consistent across folds)
- ✅ Precision/Recall balanced (not too different)
- ✅ Similar performance on CONFIRMED and FALSE POSITIVE classes

### Performance Issues
- ⚠️ Low recall on CANDIDATE class (harder to detect)
- ⚠️ High CV std indicates overfitting
- ⚠️ Large gap between train/test accuracy

---

## 🚀 Performance Notes

- **Response Time**: ~100-500ms (depending on file size)
- **File Size**: Plot is typically 200-400 KB base64-encoded
- **Caching**: Consider caching response if called frequently
- **Compression**: Use gzip compression on client side

---

## 🔗 Related Endpoints

- `POST /api/train` - Train a new model (generates visualization)
- `GET /model/status` - Check current model status
- `GET /api/download/{filename}` - Download raw model files
- `GET /stats` - Get dataset and prediction statistics

