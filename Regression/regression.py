import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import create_dataset

training_dataset, testing_dataset = create_dataset.main()

def main(training_data : pd.DataFrame, testing_data : pd.DataFrame) -> any:
    pass


def linear_regression(training_data : pd.DataFrame):
    pass


if __name__ == "__main__":
    print(training_dataset, testing_dataset)