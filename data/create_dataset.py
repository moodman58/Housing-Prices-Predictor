import pandas as pd
import re
from io import StringIO
from pathlib import Path

RAW = Path('data/raw/historical_rent_timeseries.csv')
CLEAN = Path('data/processed/historical_rent_timeseries.csv')


def main():
    # read file text and replace occurrences of ",a " with a single space
    text = RAW.read_text(encoding='utf-8')
    text = re.sub(r',a ', '', text)

    # load cleaned text into pandas
    df = pd.read_csv(StringIO(text), skipinitialspace=True, dtype=str)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # strip whitespace from string cells
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # convert numeric columns (all except first) to numbers
    if df.shape[1] > 1:
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''))

    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    long_df = convertLong(df)

    training_data, testing_data = splitData(0.8, long_df)
    print(training_data, testing_data)
    
    return df


def convertLong(df : pd.DataFrame) -> pd.DataFrame:
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


def splitData(split_index : float, df : pd.DataFrame) -> any:
    df = df.sort_values(["Date", "beds"]).reset_index(drop=True)
    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index]
    test  = df.iloc[split_index:]
    return train, test


if __name__ == "__main__":
    main()