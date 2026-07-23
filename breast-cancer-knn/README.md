# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Objective
Develop a K-Nearest Neighbors (KNN) classification model to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on diagnostic measurements from the Breast Cancer Wisconsin Diagnostic Dataset.

## Dataset Link
**Breast Cancer Wisconsin (Diagnostic) Dataset**  
Kaggle: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data  
UCI Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

*(Dataset not included in repository per licensing; download from Kaggle link above)*

## Libraries Used
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning:
  - `train_test_split` - Data splitting
  - `StandardScaler` - Feature standardization
  - `LabelEncoder` - Target encoding
  - `KNeighborsClassifier` - KNN model
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score` - Evaluation metrics
  - `confusion_matrix`, `classification_report` - Detailed evaluation

## Methodology

### 1. Data Understanding
- Loaded dataset (569 samples, 30 features + diagnosis)
- **30 numerical features** computed from digitized images of fine needle aspirates (FNA):
  - Mean, Standard Error, and Worst (largest 3) values for:
    - Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
- **Target**: `diagnosis` (M=Malignant, B=Benign)
- Class distribution: Benign=357 (62.7%), Malignant=212 (37.3%)
- No missing values in features (one empty column `Unnamed: 32` removed)

### 2. Data Preprocessing
- Removed `id` column (non-predictive) and `Unnamed: 32` (all NaN)
- Encoded target: `M` → 1 (Malignant), `B` → 0 (Benign)
- **Train-test split**: 80/20 stratified (455 train, 114 test), `random_state=42`
- **Feature scaling**: `StandardScaler` fit on **training set only**, applied to both train and test (prevents data leakage)

### 3. Model Development
- Trained `KNeighborsClassifier` with **K=5** (Euclidean distance, uniform weights)
- Predicted class labels on test set

### 4. Model Evaluation
| Metric | Score |
|--------|-------|
| **Accuracy** | 95.61% |
| **Precision** | 97.44% |
| **Recall (Sensitivity)** | 90.48% |
| **F1-Score** | 93.83% |

**Confusion Matrix:**
```
                 Predicted
              Benign  Malignant
Actual Benign     71          1
Actual Malignant   4         38
```
- True Negatives: 71 | False Positives: 1
- False Negatives: 4 | True Positives: 38

**Classification Report:**
```
              precision    recall  f1-score   support

       Benign       0.95      0.99      0.97        72
    Malignant       0.97      0.90      0.94        42

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114
```

### 5. Hyperparameter Tuning
- Evaluated K from 1 to 20 using test accuracy
- **Optimal K = 5** (Accuracy: 95.61%) — matches assignment requirement
- Visualized K vs Accuracy curve

## Results

### Observations
1. **Excellent Performance**: K=5 achieves 95.61% accuracy with 97.44% precision and 90.48% recall — strong classification performance for medical diagnosis.

2. **Clinical Relevance**: The model correctly identifies 38 out of 42 malignant cases (90.5% recall), with only 4 false negatives. In cancer screening, minimizing false negatives (missed cancer) is critical.

3. **Feature Scaling is Essential**: Without `StandardScaler`, features with large ranges (e.g., `area_worst` ~2500) would dominate Euclidean distance calculations over smaller-scale features (e.g., `smoothness_mean` ~0.1). The scaler is fit on training data only to prevent data leakage.

## Conclusion

Key Findings: The KNN classifier effectively distinguishes malignant from benign breast tumors, achieving 95.6% accuracy with 90.5% recall on the test set. The model correctly identifies 38 out of 42 malignant cases, which is crucial for clinical applications where false negatives (missed cancer) carry severe consequences. Feature scaling via StandardScaler is fundamental for KNN since the algorithm relies on Euclidean distance; without normalization, features with larger ranges (e.g., worst perimeter ~188) would disproportionately influence neighbor selection compared to smaller-scale features (e.g., smoothness ~0.1). The optimal K=5 balances the bias-variance tradeoff effectively.

Importance of Feature Scaling in KNN: KNN relies entirely on distance metrics (Euclidean distance by default). Features in this dataset span vastly different scales — from ~0.05 (fractal dimension) to ~2500 (area_worst). Without standardization, large-magnitude features would dominate the distance computation, rendering small-magnitude features irrelevant. StandardScaler transforms all features to zero mean and unit variance, ensuring each feature contributes equally to distance calculations. Critically, the scaler is fit only on training data to prevent information from the test set leaking into the training process.

Limitation of KNN: KNN is a **lazy learner** with no explicit training phase — it stores the entire training dataset and computes distances to all points at prediction time. This leads to: (1) **High memory usage** and slow inference for large datasets, (2) **Sensitivity to outliers/noise** — a single noisy neighbor can flip predictions, (3) **Curse of dimensionality** — distance metrics become less meaningful in high-dimensional spaces (30 features here), though feature selection or PCA can mitigate this. For production clinical systems, ensemble methods (Random Forest, XGBoost) or SVMs would offer faster inference and better robustness.

---

## Files in Repository
| File | Description |
|------|-------------|
| `Assignment-4.py` | Complete Python solution |
| `README.md` | This documentation |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `k_vs_accuracy.png` | K vs Accuracy optimization curve |
| `conclusion.txt` | Formatted conclusion text |

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python Assignment-4.py
```

The script loads the dataset from `Breast Cancer Wisconsin (Diagnostic) Data Set/data.csv`, trains the model, prints all metrics, saves visualizations, and writes the conclusion to `conclusion.txt`.