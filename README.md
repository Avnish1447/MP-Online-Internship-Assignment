# MP Online Internship - AI/ML Assignments

This repository contains solutions for **3 AI-ML Assignments** completed as part of the MP Online Internship program (July 20-23, 2026).

## 📁 Repository Structure

```
mp-online-internship-assignments/
├── medical-insurance-prediction/          # Assignment 1: Medical Insurance Cost Prediction
│   ├── Assignment-1.py                    # Multiple Linear Regression solution
│   ├── README.md                          # Detailed documentation
│   └── actual_vs_predicted.png
│
├── customer-churn-prediction/             # Assignment 2: Customer Churn Prediction
│   ├── Assignment-2.py                    # Logistic Regression solution
│   ├── README.md                          # Detailed documentation
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── conclusion.txt
│
└── salary-prediction-polynomial/          # Assignment 3: Salary Prediction
    ├── Assignment-3.py                    # Polynomial vs Linear Regression comparison
    ├── README.md                          # Detailed documentation
    └── conclusion.txt
```

---

## 📋 Assignment Summary

| # | Topic | Algorithm | Dataset | Key Metrics |
|---|-------|-----------|---------|-------------|
| **1** | Medical Insurance Cost Prediction | Multiple Linear Regression | [Kaggle: Medical Cost Personal](https://www.kaggle.com/datasets/mirichoi0218/insurance) | R² = **0.78**, MAE = $4,181 |
| **2** | Customer Churn Prediction | Logistic Regression | [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) | Accuracy = **80.3%**, F1 = **58.6%** |
| **3** | Salary Prediction | Polynomial Regression (deg=3) vs Linear | [Kaggle: Position Salaries](https://www.kaggle.com/datasets/akram24/position-salaries) | Poly R² = **0.98** vs Linear R² = **0.84** |

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **ML Models:** LinearRegression, LogisticRegression, PolynomialFeatures + Pipeline

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run Any Assignment
```bash
# Assignment 1
cd medical-insurance-prediction
python Assignment-1.py

# Assignment 2
cd customer-churn-prediction
python Assignment-2.py

# Assignment 3
cd salary-prediction-polynomial
python Assignment-3.py
```

---

## 📊 Key Highlights

### Assignment 1: Medical Insurance
- Multiple Linear Regression with 6 features (age, sex, bmi, children, smoker, region)
- One-hot encoding for categorical variables
- Model explains 78% of variance in insurance charges
- Smoking status is the strongest predictor

### Assignment 2: Customer Churn
- Logistic Regression with 19 features after preprocessing
- Handles missing `TotalCharges` (11 values imputed with median)
- 80/20 stratified split, feature scaling applied
- Recall = 52% (identifies half of actual churners)
- Contract type, tenure, monthly charges are top predictors

### Assignment 3: Salary Prediction
- Compares Linear vs Polynomial (degree=3) Regression
- Polynomial dramatically outperforms: R² 0.98 vs 0.84
- Captures exponential salary growth at senior levels
- Demonstrates importance of model selection for non-linear relationships

---

## 📝 Submission Details

- **Deadline:** July 23, 2026, 11:59 PM IST
- **Format:** Public GitHub Repository + Google Form
- **Datasets:** Not included (licensing); Kaggle links provided in each README

---

## 👤 Author

**MP Online Internship Participant**  
*Batch: 1A*  
*Registration: IN26011857*

---

*Each assignment folder contains its own detailed README.md with methodology, results, observations, and conclusions as per the assignment requirements.*
