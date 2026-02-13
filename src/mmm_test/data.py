"""Data loading, downloading, and preprocessing."""

from pathlib import Path

import kagglehub
import pandas as pd

from mmm_test.config import DATA_DIR, DATASET_ID


def download_dataset(force: bool = False) -> Path:
    """Download the DT Mart dataset from KaggleHub if not present locally.

    Parameters
    ----------
    force : bool, optional
        If True, re-download even if data exists locally. Default is False.

    Returns
    -------
    Path
        Path to the downloaded dataset directory.
    """
    existing_files = list(DATA_DIR.glob("*.csv"))
    if existing_files and not force:
        return DATA_DIR

    dataset_path = kagglehub.dataset_download(DATASET_ID)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    source = Path(dataset_path)
    for csv_file in source.glob("*.csv"):
        destination = DATA_DIR / csv_file.name
        destination.write_bytes(csv_file.read_bytes())

    return DATA_DIR


def load_data(force_download: bool = False) -> pd.DataFrame:
    """Load the DT Mart dataset, downloading from KaggleHub if necessary.

    Parameters
    ----------
    force_download : bool, optional
        If True, re-download even if data exists locally. Default is False.

    Returns
    -------
    pd.DataFrame
        The raw dataset with validated schema.

    Raises
    ------
    FileNotFoundError
        If no CSV files are found after download.
    ValueError
        If the downloaded data fails schema validation.
    """
    data_dir = download_dataset(force=force_download)
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    df = pd.read_csv(csv_files[0])
    validate_raw_data(df)
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate the raw dataset schema and completeness.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset to validate.

    Raises
    ------
    ValueError
        If the dataset is empty or has no columns.
    """
    if df.empty:
        raise ValueError("Dataset is empty")
    if df.columns.empty:
        raise ValueError("Dataset has no columns")


def preprocess(df: pd.DataFrame, date_column: str, target_column: str) -> pd.DataFrame:
    """Preprocess raw data for MMM model fitting.

    Sorts by date, handles missing values, and ensures correct dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    date_column : str
        Name of the date column.
    target_column : str
        Name of the target variable column.

    Returns
    -------
    pd.DataFrame
        The preprocessed dataset ready for modeling.

    Raises
    ------
    ValueError
        If required columns are missing or target contains nulls after preprocessing.
    """
    missing = [col for col in [date_column, target_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    if df[target_column].isna().any():
        null_count = df[target_column].isna().sum()
        raise ValueError(
            f"Target column '{target_column}' contains {null_count} null values"
        )

    return df
