"""Data loading, downloading, and preprocessing."""

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from mmm_demo.config import DATA_DIR, DATASET_ID, DROP_CHANNELS

# Channel groups to reduce multicollinearity.
# Digital+SEM+Content.Marketing (r>0.91), Online.marketing+Affiliates (r=0.99).
CHANNEL_GROUPS: dict[str, list[str]] = {
    "TV": ["TV"],
    "Sponsorship": ["Sponsorship"],
    "Digital": ["Digital", "SEM", "Content.Marketing"],
    "Online": ["Online.marketing", "Affiliates"],
}

# MediaInvestment.csv values are in crores; multiply to get absolute rupees.
_MEDIA_SCALE_FACTOR = 1e7

# Column name mapping from MediaInvestment.csv to model column names.
_MEDIA_COLUMN_RENAME = {
    "Content Marketing": "Content.Marketing",
    "Online marketing": "Online.marketing",
}

_CHANNEL_COLUMNS = [
    "TV",
    "Digital",
    "Sponsorship",
    "Content.Marketing",
    "Online.marketing",
    "Affiliates",
    "SEM",
]

_GROUPED_CHANNEL_COLUMNS = list(CHANNEL_GROUPS.keys())


def aggregate_channel_groups(
    df: pd.DataFrame,
    groups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Sum raw channel columns into grouped channels to reduce multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the raw channel columns.
    groups : dict[str, list[str]] | None, optional
        Mapping of group name to list of raw channel column names.
        Defaults to ``CHANNEL_GROUPS``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with raw channel columns replaced by grouped columns.

    Raises
    ------
    ValueError
        If any source column listed in *groups* is missing from *df*.
    """
    if groups is None:
        groups = CHANNEL_GROUPS

    all_sources: list[str] = []
    for sources in groups.values():
        all_sources.extend(sources)
    missing = [c for c in all_sources if c not in df.columns]
    if missing:
        raise ValueError(f"Missing source channel columns: {missing}")

    df = df.copy()
    for group_name, sources in groups.items():
        df[group_name] = df[sources].sum(axis=1)

    # Drop the original raw columns that are no longer needed
    cols_to_drop = [c for c in all_sources if c not in groups]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


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
        The prepared MMM dataset (12 rows) with target, 4 grouped media channels,
        and control variables.
    """
    data_dir = download_dataset(force=force_download)
    df = load_secondfile()
    sale_days = compute_sale_days(data_dir)

    df = df.merge(sale_days, on="Date", how="left")
    df["sale_days"] = df["sale_days"].fillna(0).astype(int)
    df = df.drop(columns=DROP_CHANNELS, errors="ignore")
    df = aggregate_channel_groups(df)

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
        *_GROUPED_CHANNEL_COLUMNS,
        "NPS",
        "total_Discount",
        "sale_days",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["total_gmv"].isna().any():
        raise ValueError("Target column 'total_gmv' contains null values")

    channel_cols = _GROUPED_CHANNEL_COLUMNS
    for col in channel_cols:
        if df[col].isna().any():
            raise ValueError(f"Channel column '{col}' contains null values")

    if len(df) != 12:
        raise ValueError(f"Expected 12 monthly observations, got {len(df)}")


# ---------------------------------------------------------------------------
# Weekly aggregation functions
# ---------------------------------------------------------------------------


def aggregate_weekly_gmv(data_dir: Path) -> pd.DataFrame:
    """Aggregate daily transactions from firstfile.csv into weekly totals.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory containing firstfile.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Date`` (Monday of each week),
        ``total_gmv`` and ``total_Discount``.

    Raises
    ------
    FileNotFoundError
        If firstfile.csv is not found.
    """
    filepath = data_dir / "firstfile.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"firstfile.csv not found in {data_dir}")

    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])

    # Floor each date to the Monday of its ISO week
    df["week_start"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")

    weekly = (
        df.groupby("week_start")
        .agg(total_gmv=("gmv_new", "sum"), total_Discount=("discount", "sum"))
        .reset_index()
        .rename(columns={"week_start": "Date"})
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return weekly


def distribute_media_to_weekly(data_dir: Path, weekly_dates: pd.Series) -> pd.DataFrame:
    """Pro-rata distribute monthly media spend across weeks.

    Each week is assigned to the month that contains its Monday date.
    Monthly spend is divided equally across all weeks in that month.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory containing MediaInvestment.csv.
    weekly_dates : pd.Series
        Series of weekly Monday dates to distribute spend into.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Date`` + 7 channel spend columns.

    Raises
    ------
    FileNotFoundError
        If MediaInvestment.csv is not found.
    """
    filepath = data_dir / "MediaInvestment.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"MediaInvestment.csv not found in {data_dir}")

    media = pd.read_csv(filepath)
    # Fix column names: strip whitespace, rename to match model conventions
    media.columns = media.columns.str.strip()
    media = media.rename(columns=_MEDIA_COLUMN_RENAME)

    # Build a month timestamp from Year + Month columns
    media["month_start"] = pd.to_datetime(
        media["Year"].astype(str) + "-" + media["Month"].astype(str) + "-01"
    )

    # Scale to absolute values and select only the 7 usable channels
    channels = [c for c in _CHANNEL_COLUMNS if c in media.columns]
    for ch in channels:
        media[ch] = (
            pd.to_numeric(media[ch], errors="coerce").fillna(0) * _MEDIA_SCALE_FACTOR
        )

    # Assign each week to its month (by Monday date)
    weeks = pd.DataFrame({"Date": weekly_dates})
    weeks["month_start"] = weeks["Date"].dt.to_period("M").dt.to_timestamp()

    # Count weeks per month to compute equal split
    weeks_per_month = weeks.groupby("month_start").size().rename("n_weeks")
    media = media.merge(weeks_per_month, on="month_start", how="left")

    # Divide monthly spend by number of weeks in that month
    for ch in channels:
        media[ch] = np.where(media["n_weeks"] > 0, media[ch] / media["n_weeks"], 0)

    # Merge weekly spend back onto the week dates
    result = weeks.merge(
        media[["month_start", *channels]], on="month_start", how="left"
    )
    result = result.drop(columns=["month_start"])

    # Fill any channels that had no media data with 0
    for ch in channels:
        result[ch] = result[ch].fillna(0)

    return result


def distribute_nps_to_weekly(data_dir: Path, weekly_dates: pd.Series) -> pd.DataFrame:
    """Assign monthly NPS scores to weeks by month membership.

    Each week receives the NPS score of the month its Monday falls in.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory containing MonthlyNPSscore.csv.
    weekly_dates : pd.Series
        Series of weekly Monday dates.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Date`` and ``NPS`` columns.

    Raises
    ------
    FileNotFoundError
        If MonthlyNPSscore.csv is not found.
    """
    filepath = data_dir / "MonthlyNPSscore.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"MonthlyNPSscore.csv not found in {data_dir}")

    nps = pd.read_csv(filepath)
    nps["Date"] = pd.to_datetime(nps["Date"])
    nps["month_start"] = nps["Date"].dt.to_period("M").dt.to_timestamp()

    weeks = pd.DataFrame({"Date": weekly_dates})
    weeks["month_start"] = weeks["Date"].dt.to_period("M").dt.to_timestamp()

    result = weeks.merge(nps[["month_start", "NPS"]], on="month_start", how="left")
    return result[["Date", "NPS"]]


def compute_weekly_sale_days(data_dir: Path, weekly_dates: pd.Series) -> pd.DataFrame:
    """Count special sale event days per week from SpecialSale.csv.

    Parameters
    ----------
    data_dir : Path
        Path to the data directory containing SpecialSale.csv.
    weekly_dates : pd.Series
        Series of weekly Monday dates.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``Date`` and ``sale_days`` columns.

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

    # Floor each sale event date to its week's Monday
    special["week_start"] = special["Date"] - pd.to_timedelta(
        special["Date"].dt.weekday, unit="D"
    )

    sale_counts = special.groupby("week_start").size().reset_index(name="sale_days")

    weeks = pd.DataFrame({"Date": weekly_dates})
    result = weeks.merge(sale_counts, left_on="Date", right_on="week_start", how="left")
    result["sale_days"] = result["sale_days"].fillna(0).astype(int)
    result = result.drop(columns=["week_start"], errors="ignore")
    return result[["Date", "sale_days"]]


def load_mmm_weekly_data(force_download: bool = False) -> pd.DataFrame:
    """Load and prepare a weekly MMM dataset from raw data files.

    Aggregates daily transactions to weekly GMV, pro-rata distributes
    monthly media spend across weeks, assigns monthly NPS to weeks,
    and counts weekly sale event days.

    Parameters
    ----------
    force_download : bool, optional
        If True, re-download even if data exists locally. Default is False.

    Returns
    -------
    pd.DataFrame
        The prepared weekly MMM dataset (~52 rows) with target,
        7 media channels, and control variables.
    """
    data_dir = download_dataset(force=force_download)

    # Step 1: Aggregate daily transactions to weekly
    weekly_gmv = aggregate_weekly_gmv(data_dir)

    # Step 2: Distribute monthly media spend across weeks
    media = distribute_media_to_weekly(data_dir, weekly_gmv["Date"])

    # Step 3: Assign monthly NPS to weeks
    nps = distribute_nps_to_weekly(data_dir, weekly_gmv["Date"])

    # Step 4: Count sale event days per week
    sale_days = compute_weekly_sale_days(data_dir, weekly_gmv["Date"])

    # Merge all together
    df = weekly_gmv.merge(media, on="Date", how="left")
    df = df.merge(nps, on="Date", how="left")
    df = df.merge(sale_days, on="Date", how="left")

    df["sale_days"] = df["sale_days"].fillna(0).astype(int)

    # Drop boundary weeks that fall outside the monthly data coverage
    # (e.g., a Monday in June when data starts in July)
    df = df.dropna(subset=["NPS"]).reset_index(drop=True)

    # Aggregate correlated channels into groups (7 → 4)
    df = aggregate_channel_groups(df)

    validate_mmm_weekly_data(df)
    return df


def validate_mmm_weekly_data(df: pd.DataFrame) -> None:
    """Validate the prepared weekly MMM dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.

    Raises
    ------
    ValueError
        If expected columns are missing, target has nulls,
        channel columns have nulls, or row count is outside 50-53.
    """
    required_columns = [
        "Date",
        "total_gmv",
        *_GROUPED_CHANNEL_COLUMNS,
        "NPS",
        "total_Discount",
        "sale_days",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["total_gmv"].isna().any():
        raise ValueError("Target column 'total_gmv' contains null values")

    for col in _GROUPED_CHANNEL_COLUMNS:
        if df[col].isna().any():
            raise ValueError(f"Channel column '{col}' contains null values")

    if not 50 <= len(df) <= 53:
        raise ValueError(f"Expected 50-53 weekly observations, got {len(df)}")
