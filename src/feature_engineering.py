import pandas as pd

def one_hot_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One‐hot encode specified categorical columns."""
    return pd.get_dummies(df, columns=cols, dummy_na=False)

def frequency_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replace each category with its overall frequency."""
    freq = df[col].value_counts(normalize=True)
    df[col + "_freq"] = df[col].map(freq)
    return df

def label_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Simple integer mapping for a single categorical column."""
    uniques = {v: i for i, v in enumerate(df[col].dropna().unique())}
    df[col + "_lbl"] = df[col].map(lambda x: uniques.get(x, -1))
    return df

def add_datetime_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    If you have a timestamp column, extract hour/day/month.
    (In IEEE-CIS data there’s no explicit datetime, but you could parse TransactionDT.)
    """
    ts = pd.to_datetime(df[time_col], unit='s', origin='unix')
    df[time_col + "_hour"]   = ts.dt.hour
    df[time_col + "_day"]    = ts.dt.day
    df[time_col + "_month"]  = ts.dt.month
    return df

__all__ = [
    "one_hot_encode",
    "frequency_encode",
    "label_encode",
    "add_datetime_features",
]
