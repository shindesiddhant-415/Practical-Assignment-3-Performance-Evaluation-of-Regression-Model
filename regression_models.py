import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read the salary data
data = pd.read_csv("Salary_Data.csv")

print("First five records:")
print(data.head())
print("\nDataset shape:", data.shape)

X = data[["YearsExperience"]]
y = data["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print("\n", name)
    print("MAE :", round(mae, 2))
    print("MSE :", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 4))

    return [name, mae, mse, rmse, r2]


# Train Linear Regression and Ridge Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

results = []
results.append(evaluate("Linear Regression", y_test, linear_pred))
results.append(evaluate("Ridge Regression", y_test, ridge_pred))


# Gradient Descent works better when the values are scaled.
# The scaler values are calculated from the training data only.
x_train = X_train["YearsExperience"].to_numpy()
x_test = X_test["YearsExperience"].to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

x_mean = x_train.mean()
x_std = x_train.std()
y_mean = y_train.mean()
y_std = y_train.std()

x_train_scaled = (x_train - x_mean) / x_std
x_test_scaled = (x_test - x_mean) / x_std
y_train_scaled = (y_train - y_mean) / y_std

slope = 0.0
intercept = 0.0
learning_rate = 0.05
iterations = 2000
n = len(x_train_scaled)
cost_history = []

for i in range(iterations):
    predicted = slope * x_train_scaled + intercept
    error = predicted - y_train_scaled

    slope_gradient = (2 / n) * np.sum(error * x_train_scaled)
    intercept_gradient = (2 / n) * np.sum(error)

    slope = slope - learning_rate * slope_gradient
    intercept = intercept - learning_rate * intercept_gradient

    new_predicted = slope * x_train_scaled + intercept
    cost = np.mean((new_predicted - y_train_scaled) ** 2)
    cost_history.append(cost)

gd_pred_scaled = slope * x_test_scaled + intercept
gd_pred = gd_pred_scaled * y_std + y_mean
results.append(evaluate("Gradient Descent", y_test, gd_pred))

print("\nGradient Descent details")
print("Learning rate:", learning_rate)
print("Iterations:", iterations)
print("Slope:", round(slope, 4))
print("Intercept:", round(intercept, 4))
print("Initial cost:", round(cost_history[0], 6))
print("Final cost:", round(cost_history[-1], 6))


# Check the result against Scikit-learn Linear Regression
linear_scaled = LinearRegression()
linear_scaled.fit(x_train_scaled.reshape(-1, 1), y_train_scaled)
sklearn_pred_scaled = linear_scaled.predict(x_test_scaled.reshape(-1, 1))
sklearn_pred = sklearn_pred_scaled * y_std + y_mean

print("\nDifference between Gradient Descent and Scikit-learn predictions:",
      round(float(np.max(np.abs(gd_pred - sklearn_pred))), 8))

# Save the result table
output_folder = Path("outputs")
output_folder.mkdir(exist_ok=True)

results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "MSE", "RMSE", "R2"]
)
results_df.to_csv(output_folder / "regression_metrics.csv", index=False)
print("\nModel comparison:")
print(results_df.round(4).to_string(index=False))


# Graph 1: actual salary and predicted salary
plt.figure(figsize=(7, 5))
plt.scatter(y_test, linear_pred, label="Predicted salary")
low = min(y_test.min(), linear_pred.min())
high = max(y_test.max(), linear_pred.max())
plt.plot([low, high], [low, high], "--", label="Perfect prediction")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_folder / "Actual_vs_Predicted_Salary.png", dpi=160)
plt.close()

# Graph 2: cost during Gradient Descent
plt.figure(figsize=(7, 5))
plt.plot(range(1, iterations + 1), cost_history)
plt.xlabel("Iterations")
plt.ylabel("MSE (scaled values)")
plt.title("Gradient Descent Cost Reduction")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_folder / "Gradient_Descent_Cost.png", dpi=160)
plt.close()

# Graph 3: compare R2 scores
plt.figure(figsize=(7, 5))
plt.bar(results_df["Model"], results_df["R2"])
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.savefig(output_folder / "R2_Comparison.png", dpi=160)
plt.close()

# Try a sample prediction for 5 years of experience
years = 5.0
years_scaled = (years - x_mean) / x_std
salary_scaled = slope * years_scaled + intercept
salary = salary_scaled * y_std + y_mean

print("\nExample prediction")
print("Years of experience:", years)
print("Predicted salary: $", round(salary, 2))

print("\nGraphs and results have been saved in the outputs folder.")
