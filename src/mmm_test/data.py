"""Data loading, downloading, and preprocessing."""

from pathlib import Path

import kagglehub
import pandas as pd

from mmm_test.config import DATA_DIR, DATASET_ID, DROP_CHANNELS


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


def load_secondfile(force_download: bool = False) -> pd.DataFrame:
    """Load Secondfile.csv (monthly aggregated MMM-ready dataset).

    Parameters
    ----------
    force_download : bool, optional
        If True, re-download even if data exists locally. Default is False.

    Returns
    -------
    pd.DataFrame
        The monthly dataset with validated schema.

    Raises
    ------
    FileNotFoundError
        If Secondfile.csv is not found.
    """
    data_dir = download_dataset(force=force_download)
    filepath = data_dir / "Secondfile.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Secondfile.csv not found in {data_dir}")

    df = pd.read_csv(filepath)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    validate_raw_data(df)
    return df


def compute_sale_days(data_dir: Path) -> pd.DataFrame:
    """Compute number of special sale days per month from SpecialSale.csv.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory containing SpecialSale.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Date' (monthly timestamp) and 'sale_days' columns.
        Months without sale events are not included.

    Raises
    ------
    FileNotFoundError
        If SpecialSale.csv is not found.
    """
    filepath = data_dir / "SpecialSale.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"SpecialSale.csv not found in {data_dir}")

    special = pd.read_csv(filepath)
    special["Date"] = pd.to_datetime(special["Date"])

    sale_days = (
        special.groupby(special["Date"].dt.to_period("M"))
        .size()
        .reset_index(name="sale_days")
    )
    sale_days["Date"] = sale_days["Date"].dt.to_timestamp()
    return sale_days


def load_mmm_data(force_download: bool = False) -> pd.DataFrame:
    """Load and prepare the full MMM dataset with sale_days control variable.

    Loads Secondfile.csv, computes sale_days from SpecialSale.csv, merges them,
    and drops channels with excessive missing data (Radio, Other).

    Parameters
    ----------
    force_download : bool, optional
        If True, re-download even if data exists locally. Default is False.

    Returns
    -------
    pd.DataFrame
        The prepared MMM dataset (12 rows) with target, 7 media channels,
        and control variables.
    """
    data_dir = download_dataset(force=force_download)
    df = load_secondfile()
    sale_days = compute_sale_days(data_dir)

    df = df.merge(sale_days, on="Date", how="left")
    df["sale_days"] = df["sale_days"].fillna(0).astype(int)
    df = df.drop(columns=DROP_CHANNELS, errors="ignore")

    validate_mmm_data(df)
    return df


def validate_mmm_data(df: pd.DataFrame) -> None:
    """Validate the prepared MMM dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.

    Raises
    ------
    ValueError
        If expected columns are missing, target has nulls,
        channel columns have nulls, or row count is not 12.
    """
    required_columns = [
        "Date",
        "total_gmv",
        "TV",
        "Digital",
        "Sponsorship",
        "Content.Marketing",
        "Online.marketing",
        "Affiliates",
        "SEM",
        "NPS",
        "total_Discount",
        "sale_days",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["total_gmv"].isna().any():
        raise ValueError("Target column 'total_gmv' contains null values")

    channel_cols = [
        "TV",
        "Digital",
        "Sponsorship",
        "Content.Marketing",
        "Online.marketing",
        "Affiliates",
        "SEM",
    ]
    for col in channel_cols:
        if df[col].isna().any():
            raise ValueError(f"Channel column '{col}' contains null values")

    if len(df) != 12:
        raise ValueError(f"Expected 12 monthly observations, got {len(df)}")
