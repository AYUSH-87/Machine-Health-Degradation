"""Data loading, cleaning, validation, and preprocessing module for multi-sensor vibration data."""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .config import AppConfig


class PreprocessingPipeline:
    """Manages scaling and feature normalization across train and inference stages."""

    def __init__(self, scaling_method: str = "robust"):
        self.scaling_method = scaling_method
        self.scaler: Optional[TransformerMixin] = None
        self.feature_names: List[str] = []

    def _create_scaler(self) -> TransformerMixin:
        if self.scaling_method == "robust":
            return RobustScaler()
        elif self.scaling_method == "standard":
            return StandardScaler()
        elif self.scaling_method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling_method: {self.scaling_method}. Choose 'robust', 'standard', or 'minmax'.")

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler on feature DataFrame and transform."""
        self.scaler = self._create_scaler()
        self.feature_names = list(X.columns)
        scaled = self.scaler.fit_transform(X.values)
        return scaled

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform feature DataFrame using fitted scaler."""
        if self.scaler is None:
            raise RuntimeError("PreprocessingPipeline has not been fitted yet.")
        # Ensure column alignment
        missing = [col for col in self.feature_names if col not in X.columns]
        if missing:
            raise ValueError(f"Input data is missing expected feature columns: {missing}")
        ordered_X = X[self.feature_names].values
        return self.scaler.transform(ordered_X)


def load_vibration_dataset(
    file_path: str,
    sensor_columns: List[str],
    timestamp_column: Optional[str] = "timestamp",
    missing_strategy: str = "forward_fill",
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load, validate, clean, and chronologically sort a multi-sensor vibration CSV.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    sensor_columns : list of str
        Expected sensor measurement column names.
    timestamp_column : str, optional
        Name of timestamp column if present.
    missing_strategy : str
        Method for handling missing values: "forward_fill", "median", or "drop".

    Returns
    -------
    cleaned_df : pd.DataFrame
        Cleaned sensor measurements.
    timestamps : pd.Series or None
        Preserved timestamps or sample indices.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Vibration dataset file not found at: {file_path}")

    # Read CSV
    df = pd.read_csv(path)

    # 1. Validate required sensor columns exist
    missing_cols = [c for c in sensor_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at '{file_path}' is missing required sensor columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # 2. Extract and preserve timestamp or sequence order
    timestamps: Optional[pd.Series] = None
    if timestamp_column and timestamp_column in df.columns:
        # Attempt conversion to datetime
        try:
            df["_parsed_timestamp"] = pd.to_datetime(df[timestamp_column])
            # Chronological sort
            df = df.sort_values(by="_parsed_timestamp").reset_index(drop=True)
            timestamps = df[timestamp_column]
            df = df.drop(columns=["_parsed_timestamp"])
        except Exception:
            # Fallback if timestamp cannot be parsed as datetime: retain original order
            timestamps = df[timestamp_column]
    else:
        timestamps = pd.Series(np.arange(len(df)), name="sample_index")

    # 3. Deduplicate records based on sensor readings and timestamp
    subset_cols = [c for c in sensor_columns]
    if timestamp_column and timestamp_column in df.columns:
        subset_cols.append(timestamp_column)
    initial_len = len(df)
    df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    dedup_dropped = initial_len - len(df)
    if dedup_dropped > 0:
        print(f"[DataLoader] Removed {dedup_dropped} duplicate records.")

    # 4. Handle invalid values (inf, -inf)
    df[sensor_columns] = df[sensor_columns].replace([np.inf, -np.inf], np.nan)

    # 5. Handle missing values
    missing_count = df[sensor_columns].isna().sum().sum()
    if missing_count > 0:
        print(f"[DataLoader] Detected {missing_count} missing values across sensor channels.")
        if missing_strategy == "forward_fill":
            df[sensor_columns] = df[sensor_columns].ffill().bfill()
        elif missing_strategy == "median":
            for col in sensor_columns:
                df[col] = df[col].fillna(df[col].median())
        elif missing_strategy == "drop":
            df = df.dropna(subset=sensor_columns).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown missing_strategy: {missing_strategy}")

    # Re-verify no NaNs remain
    if df[sensor_columns].isna().sum().sum() > 0:
        df[sensor_columns] = df[sensor_columns].fillna(0.0)

    # Convert sensor columns to float64
    for col in sensor_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float64)

    return df, timestamps
