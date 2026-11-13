"""Dataset checks and fit-on-training-only Titanic feature engineering."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

TARGET = "Survived"
RAW_FEATURES = ("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
REQUIRED = ("PassengerId", TARGET, "Name", "Ticket", "Cabin", *RAW_FEATURES)
NUMERIC = ("Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FarePerPerson")
CATEGORICAL = ("Pclass", "Sex", "Embarked")


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate labels, IDs and raw fields; retain missing feature values for imputation."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or not frame.columns.is_unique:
        raise ValueError("Expected a nonempty DataFrame with unique columns.")
    absent = set(REQUIRED) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing Titanic columns: {sorted(absent)}")
    data = frame.loc[:, REQUIRED].copy()
    if data["PassengerId"].isna().any() or data["PassengerId"].duplicated().any():
        raise ValueError("PassengerId must be non-null and unique.")
    if data[TARGET].isna().any() or not set(data[TARGET].unique()) <= {0, 1}:
        raise ValueError("Survived must contain 0/1 labels only.")
    if data[TARGET].nunique() != 2:
        raise ValueError("Both survival classes must be represented.")
    for col in ("Pclass", "Age", "SibSp", "Parch", "Fare"):
        converted = pd.to_numeric(data[col], errors="coerce")
        if (data[col].notna() & converted.isna()).any():
            raise ValueError(f"Invalid numeric values in {col}.")
        data[col] = converted
    if not set(data["Pclass"].dropna().unique()) <= {1, 2, 3}:
        raise ValueError("Pclass must be 1, 2, or 3.")
    for col in ("Age", "SibSp", "Parch", "Fare"):
        if (data[col].dropna() < 0).any():
            raise ValueError(f"Negative values in {col}.")
    if (data["Age"].dropna() > 120).any():
        raise ValueError("Age outside expected bounds.")
    for col, values in (("Sex", {"male", "female"}), ("Embarked", {"C", "Q", "S"})):
        clean = data[col].astype("string").str.strip()
        if col == "Sex":
            clean = clean.str.lower()
        if (clean.dropna().isin(values) == False).any():
            raise ValueError(f"Invalid {col} category.")
        data[col] = clean.astype(object).where(clean.notna(), np.nan)
    return data


def load_dataset(path: str | Path) -> pd.DataFrame:
    return validate_dataset(pd.read_csv(path, encoding="utf-8-sig"))


def split_features_target(frame: pd.DataFrame):
    data = validate_dataset(frame)
    return data.loc[:, RAW_FEATURES].copy(), data[TARGET].astype(int)


class TitanicFeatureBuilder(BaseEstimator, TransformerMixin):
    """Deterministic family and fare features, also used for API inference."""

    def fit(self, X, y=None):
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(RAW_FEATURES):
            raise ValueError("Expected the seven Titanic prediction features.")
        out = X.loc[:, RAW_FEATURES].copy()
        family = out["SibSp"].astype(float) + out["Parch"].astype(float) + 1
        out["FamilySize"] = family
        out["IsAlone"] = (family == 1).astype(int)
        out["FarePerPerson"] = out["Fare"] / family.replace(0, np.nan)
        out["Pclass"] = out["Pclass"].astype("object").astype("string").astype(object)
        return out


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Fit outlier thresholds on each training fold, not the validation set."""

    def __init__(self, low: float = 0.01, high: float = 0.99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        if not 0 <= self.low < self.high <= 1:
            raise ValueError("Invalid clipping quantiles.")
        data = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(data, self.low, axis=0)
        self.upper_ = np.nanquantile(data, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, list(NUMERIC)),
        ("categorical", categorical, list(CATEGORICAL)),
    ])
