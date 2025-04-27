import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def drop_constant_columns(df: pd.DataFrame, threshold: float = 0.95, exclude: list[str] = None) -> pd.DataFrame:
    exclude = exclude or []
    to_drop = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].nunique(dropna=False) <= 1:
            to_drop.append(col)
        elif df[col].value_counts(normalize=True, dropna=False).iloc[0] > threshold:
            to_drop.append(col)
    return df.drop(columns=to_drop)

def fill_missing(df: pd.DataFrame, strategy: str = 'median', exclude: list[str] = None) -> pd.DataFrame:
    exclude = exclude or []
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude]
    cat_cols = [c for c in df.select_dtypes(include=['object','category']).columns if c not in exclude]
    if strategy == 'median':
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
    else:
        for c in num_cols:
            df[c] = df[c].fillna(df[c].mean())
    for c in cat_cols:
        df[c] = df[c].fillna('Missing')
    return df

def reduce_mem_usage(df: pd.DataFrame, exclude: list[str] = None) -> pd.DataFrame:
    exclude = exclude or []
    df = df.copy()
    for col in df.select_dtypes(include='number').columns:
        if col in exclude: 
            continue
        if pd.api.types.is_integer_dtype(df[col].dtype):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        else:
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

__all__ = [
    "load_data",
    "drop_duplicates",
    "drop_constant_columns",
    "fill_missing",
    "reduce_mem_usage",
]
