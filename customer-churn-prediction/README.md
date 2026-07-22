# Customer Churn Prediction using Logistic Regression

## Objective
Develop a Logistic Regression model to predict customer churn for a telecommunications company based on demographic information and service usage patterns.

## Dataset Link
**Telco Customer Churn Dataset**  
Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Libraries Used
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning algorithms and metrics:
  - `train_test_split` - Data splitting
  - `LabelEncoder` - Categorical encoding
  - `StandardScaler` - Feature scaling
  - `LogisticRegression` - Model training
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score` - Evaluation metrics
  - `confusion_matrix`, `classification_report` - Detailed evaluation

## Methodology

### 1. Data Understanding
- Loaded the Telco Customer Churn dataset (7,043 records, 21 features)
- Identified numerical features: `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
- Identified categorical features: `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- Target variable: `Churn` (Yes/No)

### 2. Data Preprocessing
- Converted `TotalCharges` from object to numeric (handled whitespace values)
- Filled 11 missing `TotalCharges` values with median
- Encoded target variable: `Yes` → 1, `No` → 0
- Label encoded all categorical features
- Dropped `customerID` (non-predictive identifier)
- Split data: 80% training (5,634 samples), 20% testing (1,409 samples) with stratification
- Applied StandardScaler to numerical features

### 3. Model Development
- Trained Logistic Regression with `max_iter=1000`, `random_state=42`
- Used all 19 features after preprocessing
- Predicted churn probability and class labels on test set

### 4. Model Evaluation
Evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted churners, how many actually churned
- **Recall**: Of actual churners, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 80.34% |
| Precision | 66.44% |
| Recall | 52.41% |
| F1-Score | 58.59% |

### Confusion Matrix
```
                 Predicted
              No Churn  Churn
Actual No Churn   936    99
Actual Churn      178   196
```
- True Negatives: 936
- False Positives: 99
- False Negatives: 178
- True Positives: 196

### Top 10 Most Influential Features (by coefficient magnitude)
1. **MonthlyCharges** (+1.247) - Strongest positive predictor of churn
2. **TotalCharges** (-1.231) - Higher total charges reduce churn probability
3. **PhoneService** (-0.873) - Having phone service reduces churn
4. **Contract_Month-to-month** (-0.855) - Month-to-month contracts increase churn
5. **PaperlessBilling** (+0.397) - Paperless billing increases churn
6. **tenure** (+0.302) - Longer tenure increases churn (counterintuitive after scaling)
7. **OnlineSecurity** (-0.259) - Security service reduces churn
8. **TechSupport** (-0.235) - Tech support reduces churn
9. **Dependents** (-0.229) - Having dependents reduces churn
10. **InternetService** (+0.206) - Fiber optic increases churn

### Observations
1. The model achieves 80.34% accuracy with a recall of 52.41% for the churn class, indicating it correctly identifies ~52% of actual churners — valuable for targeted retention campaigns
2. When the model predicts churn, it is correct 66.44% of the time (precision), minimizing wasted retention efforts
3. MonthlyCharges, TotalCharges, PhoneService, and Contract type are the primary churn drivers; month-to-month contracts and higher monthly charges strongly correlate with churn

## Conclusion

Key Findings: The Logistic Regression model achieves 80.34% accuracy in predicting customer churn, with a recall of 52.41% for the churn class, indicating it correctly identifies over half of actual churners. The model's precision of 66.44% means retention campaigns targeting predicted churners will be efficient. The F1-score of 58.59% reflects a reasonable balance between precision and recall.

Factors Influencing Churn: The strongest predictors are monthly charges (higher charges increase churn probability), total charges (higher lifetime value reduces churn), phone service (reduces churn), and contract type (month-to-month contracts significantly increase churn risk). Lack of value-added services like OnlineSecurity and TechSupport also drives churn. Customers with fiber optic internet and paperless billing show higher churn propensity.

Limitation: Logistic Regression assumes a linear decision boundary between classes. Customer churn behavior likely involves complex non-linear interactions (e.g., tenure × contract type × service bundles, tenure × monthly charges) that a linear model cannot capture. Tree-based ensembles (Random Forest, XGBoost) or neural networks would better model these interactions, potentially improving recall for the minority churn class while maintaining precision.

---

## Files in Repository
- `Assignment-2.py` - Complete Python solution
- `README.md` - This documentation
- `confusion_matrix.png` - Confusion matrix visualization
- `feature_importance.png` - Feature coefficient visualization
- `conclusion.txt` - Conclusion text

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python Assignment-2.py
```

Note: The dataset is loaded directly from GitHub raw URL. No local dataset file required.