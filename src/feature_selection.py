import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif


def variance_threshold_selector(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Remove features with variance below the given threshold.

    Parameters:
    df (pd.DataFrame): Input features (numeric).
    threshold (float): Features with variance <= threshold will be removed.

    Returns:
    pd.DataFrame: DataFrame with low-variance features removed.
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    keep_cols = df.columns[selector.get_support(indices=True)]
    return df[keep_cols]


def select_k_best_univariate(df: pd.DataFrame, target: pd.Series, k: int = 50) -> pd.DataFrame:
    """
    Select the top k features based on univariate statistical tests (ANOVA F-value).

    Parameters:
    df (pd.DataFrame): Feature matrix.
    target (pd.Series): Target vector.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame containing only the selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(df, target)
    keep_cols = df.columns[selector.get_support(indices=True)]
    return df[keep_cols]


def tree_based_selection(df: pd.DataFrame, target: pd.Series, model, threshold: float = None) -> pd.DataFrame:
    """
    Select features based on feature importances from a tree-based model.

    Parameters:
    df (pd.DataFrame): Feature matrix.
    target (pd.Series): Target vector.
    model: A fitted tree-based model with `feature_importances_` attribute.
    threshold (float): If provided, keep features with importance >= threshold;
                       otherwise, keep top half by importance.

    Returns:
    pd.DataFrame: DataFrame with selected features.
    """
    import numpy as np

    # Fit model if not already fitted
    if not hasattr(model, 'feature_importances_'):
        model.fit(df, target)

    importances = model.feature_importances_
    if threshold is not None:
        mask = importances >= threshold
    else:
        median_imp = np.median(importances)
        mask = importances >= median_imp

    keep_cols = df.columns[mask]
    return df[keep_cols]

__all__ = [
    'variance_threshold_selector',
    'select_k_best_univariate',
    'tree_based_selection'
]
