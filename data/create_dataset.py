import pandas as pd
import re
from pathlib import Path

RAW = r'data\raw\historical_rent_timeseries.csv'
CLEAN = r'\data\processed\historical_rent_timeseries.csv'


def main():
    df = pd.read_csv(RAW, skipinitialspace=True,)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    
    print(df)


if __name__ == "__main__":
    res = main()

