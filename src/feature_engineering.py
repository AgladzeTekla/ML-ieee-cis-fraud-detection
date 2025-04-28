import pandas as pd


def label_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Integer–encode the categories of a single column.
    """
    df = df.copy()
    uniques = df[col].dropna().unique()
    mapping = {v: i for i, v in enumerate(uniques)}
    df[col + '_lbl'] = df[col].map(lambda x: mapping.get(x, -1))
    return df


def frequency_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Replace each category by its normalized frequency.
    """
    df = df.copy()
    freq = df[col].value_counts(normalize=True)
    df[col + '_freq'] = df[col].map(lambda x: freq.get(x, 0))
    return df


def one_hot_encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    One–hot encode multiple categorical columns.
    """
    return pd.get_dummies(df, columns=cols, dummy_na=False)


def create_time_features(df: pd.DataFrame, time_col: str = 'TransactionDT') -> pd.DataFrame:
    """
    From TransactionDT (seconds since reference), extract:
      - day number
      - hour of day
      - day of week
    """
    df = df.copy()
    # days since origin
    df['Transaction_day'] = (df[time_col] // (3600 * 24)).astype(int)
    # hour within day
    df['Transaction_hour'] = ((df[time_col] % (3600 * 24)) // 3600).astype(int)
    # approximate weekday (mod 7)
    df['Transaction_weekday'] = (df['Transaction_day'] % 7).astype(int)
    return df

__all__ = [
    'label_encode',
    'frequency_encode',
    'one_hot_encode',
    'create_time_features',
]
