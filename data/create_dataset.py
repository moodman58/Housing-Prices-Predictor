import pandas as pd
import numpy as np
import re
from io import StringIO
from pathlib import Path

RAW = Path('data/raw/historical_rent_timeseries.csv')
CLEAN = Path('data/processed/historical_rent_timeseries.csv')


def main():

    # read file text and replace occurrences of ",a " with a single space
    text = RAW.read_text(encoding='utf-8')
    text = re.sub(r',a ', '', text)

    # Read CSV
    df = pd.read_csv(StringIO(text), skipinitialspace=True, dtype=str)

    # Clean up  
    df = cleanDataFrame(df=df)

    # Cleaned up CSV file 
    # CLEAN.parent.mkdir(parents=True, exist_ok=True)
    # df.to_csv(CLEAN, index=False)
    
    training_data, testing_data = createDataset(df=df)

    return training_data, testing_data


def removeCommas(df : pd.DataFrame) -> pd.DataFrame:
    """Remove commas from numbers in all columns except first column"""

    if df.shape[1] > 1:
        for col in df.columns[1:]: 
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''))
    return df


def cleanDataFrame(df : pd.DataFrame) -> pd.DataFrame:
    """Applies methods in order that clean up the dataframe"""

    df = removeUnnamed(df)
    df = removeCommas(df)
    return df


def createDataset(df : pd.DataFrame) -> pd.DataFrame:
    """Gets the dataframe to a ready state to apply ML algorithms"""
    
    training_data_set = {}
    testing_data_set = {}

    # Convert to long DF for easier access to data
    long_df = convertLong(df)

    # Split data, currently have a 80-20 split between training and testing data
    training_data, testing_data = splitData(0.8, long_df)

    training_data = training_data.sort_values(["Date", "beds"])

    # Split training and testing data by bedroom, and convert values to np arrays
    for column, row in training_data.groupby("beds", sort=False):
        if column not in training_data_set:
            training_data_set[column] = []
        training_data_set[column].append(row["avg_rent"].to_numpy()) 

    for column, row in testing_data.groupby("beds", sort=False):
        if column not in testing_data_set:
            testing_data_set[column] = []
        testing_data_set[column].append(row["avg_rent"].to_numpy()) 

    return training_data_set, testing_data_set


def convertLong(df : pd.DataFrame) -> pd.DataFrame:
    """
    Converts df to a long df, in this case just converts all different bedroom columns to one column under 'beds'
    with avg_rent being the avg_rent for that particularly bed, instead of the avg between all those bed choices for that year
    """

    rent_cols = [c for c in df.columns if (c != "Date" and c != "Total")]

    long = df.melt(
        id_vars=["Date"],
        value_vars=rent_cols,
        var_name="beds",
        value_name="avg_rent",
    )

    long = long.dropna(subset=["avg_rent"]).sort_values(["beds", "Date"])
    long["avg_rent"] = long["avg_rent"].astype(float)
    
    return long


def splitData(split_index : float, df : pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame]:
    """Splits data in to Training and Testing data, splitting bed types evenly to prevent bias"""

    df = df.sort_values(["Date", "beds"]).reset_index(drop=True)
    split_index = int(len(df) * 0.8)

    train = df.iloc[:split_index]
    test  = df.iloc[split_index:]

    return train, test


def removeUnnamed(df : pd.DataFrame) -> pd.DataFrame:
    """Removes Unnamed columns from df"""
    return df.loc[:, ~df.columns.str.match(r"^Unnamed")]

if __name__ == "__main__":
    
    main()