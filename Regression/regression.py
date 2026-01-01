import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data import create_dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

training_dataset, testing_dataset, time_dataset = create_dataset.main()


def main(training_data : pd.DataFrame, testing_data : pd.DataFrame) -> any:
    pass


def linear_regression_train_test(
    dataset: pd.DataFrame,
    time_dataset: pd.Series,
    target_col: str = "1 Bedroom",
    test_frac: float = 0.2,
    plot: bool = True
):
    # --- build X (days since t0) ---
    t0 = time_dataset.min()
    X_all = (time_dataset - t0).dt.days.to_numpy().reshape(-1, 1)
    y_all = dataset[target_col]

    # --- time-based split (NO shuffle) ---
    n = len(X_all)
    split = int(n * (1 - test_frac))

    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    # --- train on training data only ---
    model = LinearRegression()
    model.fit(X_train, y_train)

    # --- predict on both (for metrics / plotting) ---
    yhat_train = model.predict(X_train)
    yhat_test  = model.predict(X_test)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    # Can turn on and off in function argument
    if plot:
        # Plot train/test points + fitted line
        plt.figure(figsize=(10,6))
        plt.scatter(X_train, y_train, label="Train", alpha=0.7)
        plt.scatter(X_test, y_test, label="Test", alpha=0.7)
        # line across full range
        plt.plot(X_all, model.predict(X_all), linewidth=2, label="Regression Line")
        plt.title(f"{target_col} Condo Rent Prices in Toronto (Train/Test Split)")
        plt.xlabel("Days since start date")
        plt.ylabel("Avg Rent Price")
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot test actual vs predicted (easy to read)
        plt.figure(figsize=(10,4))
        plt.plot(time_dataset.iloc[split:], y_test, label="Actual (Test)")
        plt.plot(time_dataset.iloc[split:], yhat_test, label="Predicted (Test)")
        plt.title("Test: Actual vs Predicted")
        plt.grid(True)
        plt.legend()
        plt.show()

    return {
        "slope": slope,
        "intercept": intercept,
        "model": model,
        "t0": t0,
        "split_index": split,
        "yhat_test": yhat_test,
        "y_test": y_test,
        "y_train": y_train,
    }


def linear_regression_metrics(y_test : pd.Series, yhat_test : np.ndarray ) -> list:
    """useful metrics to evaluate the performance of a model"""
    mae  = mean_absolute_error(y_test, yhat_test)
    rmse = np.sqrt(mean_squared_error(y_test, yhat_test))
    r2   = r2_score(y_test, yhat_test)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def linear_regression_baseline(y_test : pd.Series, y_train : pd.Series) -> dict:
    """baseline model that juse uses last results to test our model against"""
    
    yhat_last = np.full(shape=len(y_test), fill_value=y_train[-1], dtype=float)
    metrics = linear_regression_metrics(y_test, yhat_last)

    mae_l, rmse_l, r2_l = metrics["mae"], metrics["rmse"], metrics["r2"]

    print("Baseline (last value):", mae_l, rmse_l, r2_l)
    
    return {
        "mae_baseline": mae_l,
        "rmse_basline": rmse_l,
        "r2_baseline": r2_l
        }


if __name__ == "__main__":

    results = linear_regression_train_test(training_dataset, time_dataset, target_col="1 Bedroom", test_frac=0.2)
    
    y_train = results["y_train"]
    y_test = results["y_test"]

    baseline_model_results = linear_regression_baseline(y_test=y_test, y_train=y_train)
    created_model_results = linear_regression_metrics(y_test=y_test, yhat_test=results["yhat_test"])
    mae_created, rmse_created, r2_created = created_model_results['mae'], created_model_results['rmse'], created_model_results['r2']

    print("Created:", mae_created, rmse_created, r2_created)
