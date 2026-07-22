# Medical Insurance Cost Prediction

## Objective
The objective of this project is to develop a Multiple Linear Regression model that estimates the medical insurance charges of customers based on their personal and health-related information.

## Dataset Link
[Medical Cost Personal Insurance Dataset on Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)

*(Note: The dataset is not included in this repository to comply with licensing and storage best practices. Please download it from the link above.)*

## Libraries Used
- **Pandas**: For data loading, manipulation, and preprocessing.
- **NumPy**: For numerical computations.
- **Scikit-Learn**: For dataset splitting (`train_test_split`), building the machine learning model (`LinearRegression`), and evaluating its performance (`mean_absolute_error`, `mean_squared_error`, `r2_score`).
- **Matplotlib**: For creating data visualizations (scatter plot of Actual vs Predicted charges).

## Methodology
1. **Data Understanding**: Loaded the dataset using pandas, displayed the top records, and identified the categorical attributes (`sex`, `smoker`, `region`), numerical attributes (`age`, `bmi`, `children`), and the target variable (`charges`).
2. **Data Preprocessing**: Checked for any missing values across all columns. Transformed the categorical variables into a numerical format utilizing one-hot encoding (`pd.get_dummies`).
3. **Data Splitting**: Divided the processed dataset into a training set (80%) to train the model, and a testing set (20%) to evaluate it.
4. **Model Development**: Initialized a Multiple Linear Regression model from Scikit-Learn and trained it using the multiple independent variables to predict the numerical target.
5. **Model Evaluation**: Tested the model on the unseen testing data. Quantified the performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score. Plotted Actual vs. Predicted values to visually evaluate model performance.

## Results
- **Mean Absolute Error (MAE)**: 4181.19
- **Mean Squared Error (MSE)**: 33596915.85
- **R² Score**: 0.7836

### Key Observations:
1. The **R² score (0.7836)** indicates that our model explains a significant portion of the variance in medical charges, proving it to be a reasonably robust base predictor.
2. The **scatter plot** reveals that while a large portion of predictions clusters around the ideal fit line, there is a distinct subgroup of high-charge outliers where the model systematically underestimates the true costs.
3. Linear relationships capture the overall data trends, but the presence of categorical splits (most notably, smoking status) creates separate strata in the distribution that a single generalized linear slope captures imperfectly.

## Conclusion
This project successfully built a Multiple Linear Regression model to predict medical insurance costs. Key findings demonstrate that while the model captures the overall trend, there remains significant variance it cannot explain perfectly. The most prominent factors affecting insurance charges appear tightly connected to features like smoking status, age, and BMI. 
One notable limitation of Linear Regression for this specific problem is its strict assumption of linear relations across variables; real-world health risks often amplify non-linearly (for instance, the compounding effect of age and high BMI in smokers). Furthermore, the model tends to systematically underestimate medical costs for the most extreme instances, suggesting that advanced and complex nonlinear models (e.g., Random Forests or Gradient Boosting) might capture these underlying dependencies far better than a single hyperplane.
