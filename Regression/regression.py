import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from data import create_dataset

training_dataset, testing_dataset, time_dataset = create_dataset.main()


def main(training_data : pd.DataFrame, testing_data : pd.DataFrame) -> any:
    pass


def linear_regression(training_dataset : pd.DataFrame, time_dataset : np.ndarray):
    """Basic Simple Linear Regression"""

    t0 = time_dataset.min()
    Y = training_dataset['1 Bedroom']
    X = (time_dataset - t0).dt.days.to_numpy().reshape(-1, 1)  # numeric feature

    model = LinearRegression()
    model.fit(X, Y) 
    Y_pred = model.predict(X)

    plt.figure(figsize=(8,6)) 
    plt.scatter(X, Y, color='blue', label='Data Points') 
    plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line') 
    plt.title('Linear Regression on Random Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
   linear_regression(training_dataset=training_dataset, time_dataset=time_dataset) 