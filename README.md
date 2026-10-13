# Salary Prediction Using Regression Models and Gradient Descent

## Aim
Predict salary from years of experience, evaluate Regression models using MAE, MSE, RMSE and R², and implement Gradient Descent from scratch.

## Dataset
`Salary_Data.csv` is included in this project. It contains 30 sample records with `YearsExperience` as the input feature and `Salary` as the target. A commonly shared copy of this beginner dataset is available at: https://github.com/sudarshan-koirala/Salary-Prediciton-based-on-Years-of-Experience/blob/master/Salary_Data.csv

This small dataset is intended for a basic classroom regression demonstration; results should not be treated as a reliable salary guide for real hiring decisions.

## Models
- Linear Regression
- Ridge Regression
- Gradient Descent for Linear Regression, implemented from scratch

## Metrics
MAE, MSE, RMSE and R² on the held-out test set. Gradient Descent is compared with Scikit-learn Linear Regression using the same train/test split.

## Run
```bash
pip install -r requirements.txt
python regression_models.py
```

The script saves output tables and plots in `outputs/`. Use the values from your own run in the practical report.
