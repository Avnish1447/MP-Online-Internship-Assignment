import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Task 1: Data Understanding
print("--- Task 1: Data Understanding ---")
# Load the dataset
df = pd.read_csv('insurance.csv')

# Display the first five records
print("First five records:")
print(df.head())

# Identify features and target
print("\nNumerical Features: age, bmi, children")
print("Categorical Features: sex, smoker, region")
print("Target Variable: charges\n")

# Task 2: Data Preprocessing
print("--- Task 2: Data Preprocessing ---")
# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Encode categorical variables using one-hot encoding or label encoding
# We'll use pd.get_dummies to encode categorical variables (sex, smoker, region)
# Alternatively, LabelEncoder can be used. Since they are nominal, get_dummies is good.
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
print("\nData after encoding categorical variables:")
print(df_encoded.head())

# Define Features (X) and Target (y)
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}\n")


# Task 3: Model Development
print("--- Task 3: Model Development ---")
# Build and train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the charges for the test dataset
y_pred = model.predict(X_test)
print("Model training completed and predictions made on the test set.\n")

# Task 4: Model Evaluation
print("--- Task 4: Model Evaluation ---")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Create Actual vs Predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Insurance Charges')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
print("\nPlot saved as 'actual_vs_predicted.png'")

# Observations
print("\nObservations:")
print("1. The R² score indicates that our model explains a significant portion of the variance in medical charges.")
print("2. The scatter plot shows predictions cluster around the ideal red line, but there are distinct segments of high-charge outliers where the model underestimates the costs.")
print("3. Linear relationships capture general trends, but we can see some distinct grouping/clusters in the actual charges likely due to variables like smoking.")

# Task 5: Conclusion
print("\n--- Task 5: Conclusion ---")
conclusion = """
Conclusion:
This analysis successfully built a Multiple Linear Regression model to predict medical insurance costs. 
Key findings show that while the model captures the overall trend, there remains significant variance it cannot explain perfectly. 
The most prominent factors affecting insurance charges are closely tied to features like smoking status, age, and BMI. 
One notable limitation of Linear Regression for this specific problem is its assumption of a strictly linear relationship; 
health risks often grow non-linearly (e.g., the compounding effects of age and BMI in smokers). Furthermore, the model tends to 
underestimate medical costs for extreme cases, suggesting that more complex models (e.g., Random Forests) might capture 
these nuances better than a simple line.
"""
print(conclusion)
