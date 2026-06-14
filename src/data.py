"""
Data loading and preprocessing for the credit-rating thesis.

This is a 1:1 refactor of the preprocessing block from the original Colab
script, broken into named functions. The logic and ordering are unchanged:

    load -> collapse notched ratings -> winsorize features
         -> encode ratings as integers -> leakage-safe group split -> save
"""

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src import config


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw credit-rating CSV."""
    path = path or config.DATA_PATH
    return pd.read_csv(path)


def collapse_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse notched ratings to broad classes (AAA-, AAA, AAA+ -> AAA).

    Done by stripping every '+' and '-' from the rating string, exactly as
    in the original script.
    """
    df = df.copy()
    df[config.RATING_COL] = (
        df[config.RATING_COL]
        .str.replace("+", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    return df


def winsorize(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Clip each feature to its 1st/99th percentile to tame outliers."""
    df = df.copy()
    cols = cols or config.FEATURE_COLS
    for col in cols:
        lo = df[col].quantile(config.WINSOR_LO)
        hi = df[col].quantile(config.WINSOR_HI)
        df[col] = df[col].clip(lo, hi)
    return df


def encode_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Map broad ratings to consecutive integers (D=0 ... AAA=9)."""
    df = df.copy()
    rating_order = [r for r in config.RATING_ORDER if r in df[config.RATING_COL].unique()]
    rating_map = {r: i for i, r in enumerate(rating_order)}
    df[config.TARGET_COL] = df[config.RATING_COL].map(rating_map)
    return df


def group_split(df: pd.DataFrame):
    """Company-aware train/test split.

    A company that appears many times (rated repeatedly / by several agencies)
    is kept entirely in either train OR test, never both. This prevents the
    data leakage described in the thesis. Returns X_train, X_test, y_train, y_test.
    """
    X = df[config.FEATURE_COLS]
    y = df[config.TARGET_COL]
    groups = df[config.GROUP_COL]

    gss = GroupShuffleSplit(
        n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test


def build_splits(path=None, save=True, verbose=True):
    """Run the full preprocessing pipeline and (optionally) save the splits.

    Returns X_train, X_test, y_train, y_test.
    """
    df = load_raw(path)
    df = collapse_ratings(df)
    df = winsorize(df)
    df = encode_ratings(df)

    if verbose:
        print("Unique ratings after collapsing:",
              sorted(df[config.RATING_COL].unique()))
        print(f"Unmapped ratings: {df[config.TARGET_COL].isna().sum()}")

    X_train, X_test, y_train, y_test = group_split(df)

    if save:
        joblib.dump((X_train, X_test, y_train, y_test), config.SPLITS_PATH)
        if verbose:
            print(f"Saved splits -> {config.SPLITS_PATH}")

    if verbose:
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        print(f"Classes in train: {sorted(y_train.unique())}")
        print(f"Classes in test:  {sorted(y_test.unique())}")

    return X_train, X_test, y_train, y_test


def load_splits(path=None):
    """Load the saved (X_train, X_test, y_train, y_test) tuple."""
    path = path or config.SPLITS_PATH
    return joblib.load(path)
