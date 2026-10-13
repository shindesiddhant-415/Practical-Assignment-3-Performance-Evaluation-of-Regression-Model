"""
Practical Assignment: Salary Prediction Using Regression Models and Gradient Descent

Task 1: Train regression models and evaluate MAE, MSE, RMSE and R2.
Task 2: Implement Gradient Descent for Linear Regression from scratch and study convergence.
Dataset: Salary_Data.csv (included with this project; no download required).

Run:
    pip install -r requirements.txt
    python regression_models.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def evaluate_model(name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    metrics = {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(actual, predicted),
    }
    print(f"\n{name}\n" + "-" * len(name))
    for metric in ["MAE", "MSE", "RMSE", "R2"]:
        print(f"{metric:5s}: {metrics[metric]:.4f}")
    return metrics


def gradient_descent(x, y_scaled, learning_rate, iterations):
    """Batch Gradient Descent for y_scaled_hat = slope*x_scaled + intercept."""
    slope = 0.0
    intercept = 0.0
    n = len(x)
    costs = []
    for _ in range(iterations):
        pred = slope * x + intercept
        error = pred - y_scaled
        slope_gradient = (2 / n) * np.sum(error * x)
        intercept_gradient = (2 / n) * np.sum(error)
        slope -= learning_rate * slope_gradient
        intercept -= learning_rate * intercept_gradient
        costs.append(float(np.mean((slope * x + intercept - y_scaled) ** 2)))
    return slope, intercept, costs


def main():
    # 1. Load dataset
    csv_path = Path("Salary_Data.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Salary_Data.csv must be in the same folder as regression_models.py")
    data = pd.read_csv(csv_path)
    print("First five records:")
    print(data.head())
    print("\nDataset shape:", data.shape)
    print("Missing values:", int(data.isna().sum().sum()))

    X = data[["YearsExperience"]]
    y = data["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # 2. Train the two conventional regression models
    linear_model = make_pipeline(StandardScaler(), LinearRegression())
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)

    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)

    rows = []
    linear_metrics = evaluate_model("Linear Regression", y_test, linear_pred)
    ridge_metrics = evaluate_model("Ridge Regression", y_test, ridge_pred)
    rows.extend([linear_metrics, ridge_metrics])

    # 3. Prepare one-feature data for Gradient Descent. Scale x and y from training data only.
    x_train = X_train["YearsExperience"].to_numpy(dtype=float)
    x_test = X_test["YearsExperience"].to_numpy(dtype=float)
    y_train_arr = y_train.to_numpy(dtype=float)
    y_test_arr = y_test.to_numpy(dtype=float)
    x_mean, x_std = x_train.mean(), x_train.std()
    y_mean, y_std = y_train_arr.mean(), y_train_arr.std()
    x_train_s = (x_train - x_mean) / x_std
    x_test_s = (x_test - x_mean) / x_std
    y_train_s = (y_train_arr - y_mean) / y_std

    learning_rate = 0.05
    iterations = 2000
    slope, intercept, cost_history = gradient_descent(
        x_train_s, y_train_s, learning_rate, iterations
    )
    gd_test_scaled = slope * x_test_s + intercept
    gd_pred = gd_test_scaled * y_std + y_mean
    gd_metrics = evaluate_model("Gradient Descent Linear Regression", y_test_arr, gd_pred)
    rows.append(gd_metrics)

    print("\nGradient Descent Parameters")
    print("Learning rate:", learning_rate)
    print("Iterations:", iterations)
    print("Slope (standardized values):", round(slope, 6))
    print("Intercept (standardized target):", round(intercept, 6))
    print("Initial training MSE (standardized target):", round(cost_history[0], 6))
    print("Final training MSE (standardized target):", round(cost_history[-1], 6))
    print("Cost decreased:", cost_history[-1] < cost_history[0])

    # 4. Compare GD with sklearn Linear Regression on the same standardized x and y.
    sklearn_scaled_y_model = LinearRegression()
    sklearn_scaled_y_model.fit(x_train_s.reshape(-1, 1), y_train_s)
    sklearn_pred_scaled = sklearn_scaled_y_model.predict(x_test_s.reshape(-1, 1))
    sklearn_pred = sklearn_pred_scaled * y_std + y_mean
    sklearn_gd_metrics = evaluate_model(
        "Scikit-learn Linear Regression (same feature)", y_test_arr, sklearn_pred
    )
    print("\nMaximum absolute difference in test predictions:",
          round(float(np.max(np.abs(gd_pred - sklearn_pred))), 8))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)
    pd.DataFrame({"Iteration": np.arange(1, iterations + 1),
                  "Training_MSE_Standardized": cost_history}).to_csv(
        OUTPUT_DIR / "gradient_descent_history.csv", index=False
    )

    # 5. Actual vs predicted graph for Multiple Linear / simple feature model
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, linear_pred, label="Predicted salary")
    low = min(float(y_test.min()), float(np.min(linear_pred)))
    high = max(float(y_test.max()), float(np.max(linear_pred)))
    plt.plot([low, high], [low, high], "--", label="Perfect prediction")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Actual_vs_Predicted_Salary.png", dpi=160)
    plt.close()

    # 6. Gradient Descent cost curve
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, iterations + 1), cost_history)
    plt.xlabel("Iterations")
    plt.ylabel("MSE (standardized target)")
    plt.title("Gradient Descent Cost Reduction")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Gradient_Descent_Cost.png", dpi=160)
    plt.close()

    # 7. R2 comparison graph
    model_names = metrics_df["Model"].tolist()
    r2_values = metrics_df["R2"].tolist()
    plt.figure(figsize=(8, 5))
    plt.bar(model_names, r2_values)
    plt.ylabel("R² Score")
    plt.title("R² Score Comparison of Regression Models")
    plt.xticks(rotation=18, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "R2_Comparison.png", dpi=160)
    plt.close()

    # 8. Learning-rate experiment, with standardized x and y
    rate_rows = []
    for rate in [0.001, 0.01, 0.05, 0.5, 1.05]:
        with np.errstate(over="ignore", invalid="ignore"):
            _, _, history = gradient_descent(x_train_s, y_train_s, rate, 200)
        final_cost = history[-1]
        status = "Diverged" if (not np.isfinite(final_cost) or final_cost > 1e6) else "Cost reduced / stable"
        rate_rows.append({"Learning_Rate": rate,
                          "Initial_MSE": history[0],
                          "Final_MSE": final_cost,
                          "Status": status})
    pd.DataFrame(rate_rows).to_csv(OUTPUT_DIR / "learning_rate_experiment.csv", index=False)
    print("\nLearning-rate experiment:")
    print(pd.DataFrame(rate_rows).round(6).to_string(index=False))

    # 9. Example prediction for a user-entered experience value
    example_years = 5.0
    example_scaled = (example_years - x_mean) / x_std
    example_target_scaled = slope * example_scaled + intercept
    example_salary = example_target_scaled * y_std + y_mean
    print(f"\nExample: predicted salary for {example_years:.1f} years of experience = ${example_salary:,.2f}")

    print("\nResults saved in:", OUTPUT_DIR.resolve())
    print("Use values from this run when filling in the report.")

if __name__ == "__main__":
    main()
