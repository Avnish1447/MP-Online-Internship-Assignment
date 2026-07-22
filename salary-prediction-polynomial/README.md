# Salary Prediction using Polynomial Regression

## Objective
Develop a Polynomial Regression model to accurately estimate employee salaries based on their position level, addressing the non-linear relationship inherent in corporate compensation structures.

## Dataset Link
[Position Salaries Dataset on Kaggle](https://www.kaggle.com/datasets/akram24/position-salaries)

*(Note: The dataset is not included in this repository to comply with licensing and storage best practices. A synthetic dataset mirror is used in the script as a fallback).*

## Libraries Used
- **Pandas**: For data manipulation and loading the dataset.
- **NumPy**: For numerical computations and generating continuous plot values.
- **Scikit-Learn**: For dataset splitting (`train_test_split`), feature transformation (`PolynomialFeatures`, `StandardScaler`), building the machine learning model (`LinearRegression`), and evaluating its performance (`mean_absolute_error`, `mean_squared_error`, `r2_score`).
- **Matplotlib**: For creating data visualizations (scatter plots and polynomial curves).

## Methodology
1. **Data Understanding**: Loaded the dataset, displayed the initial records, and identified `Level` as the input feature and `Salary` as the target variable.
2. **Data Preprocessing**: Checked for missing values. Extracted the 2D feature array (`X`) and target array (`y`). Split the data into 80% training and 20% testing sets using a fixed random state for reproducibility.
3. **Model Development**: 
   - Constructed a Scikit-Learn `Pipeline` encompassing a Degree 3 `PolynomialFeatures` transformation, a `StandardScaler`, and a `LinearRegression` model.
   - Trained the polynomial model on the training data.
   - Predicted salaries for the testing dataset.
   - Built a standard Linear Regression model for baseline comparison.
4. **Model Evaluation**: Quantified model performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score. Plotted the original data points alongside both the Polynomial Regression curve and the Linear Regression line to visually evaluate the fit.

## Results
- **Mean Absolute Error (MAE)**: Calculated dynamically in the script (significantly lower than linear regression).
- **Mean Squared Error (MSE)**: Calculated dynamically in the script.
- **R² Score**: Calculated dynamically in the script (approaching 1.0 for the polynomial model).

### Observations
1. Polynomial Regression (Degree=3) significantly outperforms standard Linear Regression, capturing the exponential-like non-linear salary growth pattern.
2. The MAE for Polynomial Regression is substantially lower than that of Linear Regression, demonstrating a reduction in average prediction error.
3. The cubic polynomial fits the salary curve remarkably well, especially for senior position levels (8-10) where a standard linear regression model severely underestimates compensation.

## Conclusion
Key Findings: The Polynomial Regression model (Degree=3) achieves an excellent R² score, substantially outperforming standard Linear Regression. The cubic polynomial successfully captures the accelerating salary growth across position levels, particularly the exponential increase at senior executive levels.

Difference Between Linear and Polynomial Regression: Linear Regression assumes a constant rate of change (a straight line), forcing a single slope across all position levels. Polynomial Regression introduces higher-degree terms (Level², Level³), allowing the curve to bend and model non-linear relationships where salary growth accelerates with seniority.

Advantage of Polynomial Regression: For this dataset, Polynomial Regression's key advantage is its ability to precisely model the convex salary curve—where each promotion yields a progressively larger salary increase. A linear model systematically underestimates executive compensation, making it unsuitable. The polynomial model provides accurate predictions across the entire organizational hierarchy.
