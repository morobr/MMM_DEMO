"""Tests for data module."""

import numpy as np
import pandas as pd
import pytest

from mmm_test.data import (
    aggregate_channel_groups,
    aggregate_weekly_gmv,
    compute_sale_days,
    compute_weekly_sale_days,
    distribute_media_to_weekly,
    distribute_nps_to_weekly,
    preprocess,
    validate_mmm_data,
    validate_mmm_weekly_data,
    validate_raw_data,
)


def test_validate_raw_data_passes(sample_dataframe):
    """Test validation passes for valid data."""
    validate_raw_data(sample_dataframe)


def test_validate_raw_data_empty():
    """Test validation fails for empty DataFrame."""
    with pytest.raises(ValueError, match="empty"):
        validate_raw_data(pd.DataFrame())


def test_preprocess_sorts_by_date(sample_dataframe):
    """Test that preprocessing sorts by date."""
    shuffled = sample_dataframe.sample(frac=1, random_state=42)
    result = preprocess(shuffled, "date", "sales")
    assert result["date"].is_monotonic_increasing


def test_preprocess_converts_date(sample_dataframe):
    """Test that preprocessing converts date column to datetime."""
    df = sample_dataframe.copy()
    df["date"] = df["date"].astype(str)
    result = preprocess(df, "date", "sales")
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_preprocess_missing_column(sample_dataframe):
    """Test that preprocessing raises on missing columns."""
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess(sample_dataframe, "nonexistent", "sales")


def test_preprocess_null_target(sample_dataframe):
    """Test that preprocessing raises when target has nulls."""
    df = sample_dataframe.copy()
    df.loc[0, "sales"] = None
    with pytest.raises(ValueError, match="null values"):
        preprocess(df, "date", "sales")


def test_compute_sale_days(tmp_path):
    """Test sale_days computation from SpecialSale.csv."""
    special_csv = tmp_path / "SpecialSale.csv"
    special_csv.write_text(
        "Date,Sales Name\n7/18/2015,Sale A\n7/19/2015,Sale A\n8/15/2015,Sale B\n"
    )
    result = compute_sale_days(tmp_path)
    assert "sale_days" in result.columns
    assert "Date" in result.columns
    jul = result[result["Date"] == pd.Timestamp("2015-07-01")]
    assert jul["sale_days"].values[0] == 2
    aug = result[result["Date"] == pd.Timestamp("2015-08-01")]
    assert aug["sale_days"].values[0] == 1


def test_compute_sale_days_file_not_found(tmp_path):
    """Test that compute_sale_days raises when file is missing."""
    with pytest.raises(FileNotFoundError):
        compute_sale_days(tmp_path)


def test_validate_mmm_data_passes(mmm_sample_dataframe):
    """Test validation passes for properly structured data."""
    validate_mmm_data(mmm_sample_dataframe)


def test_validate_mmm_data_missing_column(mmm_sample_dataframe):
    """Test validation fails when a required column is missing."""
    df = mmm_sample_dataframe.drop(columns=["NPS"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_mmm_data(df)


def test_validate_mmm_data_wrong_row_count(mmm_sample_dataframe):
    """Test validation fails when row count is not 12."""
    df = mmm_sample_dataframe.head(6)
    with pytest.raises(ValueError, match="Expected 12"):
        validate_mmm_data(df)


def test_validate_mmm_data_null_target(mmm_sample_dataframe):
    """Test validation fails when target has null values."""
    df = mmm_sample_dataframe.copy()
    df.loc[0, "total_gmv"] = None
    with pytest.raises(ValueError, match="total_gmv"):
        validate_mmm_data(df)


def test_validate_mmm_data_null_channel(mmm_sample_dataframe):
    """Test validation fails when a channel column has null values."""
    df = mmm_sample_dataframe.copy()
    df.loc[0, "TV"] = None
    with pytest.raises(ValueError, match="TV"):
        validate_mmm_data(df)


# ---------------------------------------------------------------------------
# Weekly aggregation tests
# ---------------------------------------------------------------------------


def test_aggregate_weekly_gmv(tmp_path):
    """Test weekly GMV aggregation from firstfile.csv."""
    # Create a small firstfile.csv spanning 2 weeks
    rows = []
    for day in pd.date_range("2015-07-06", periods=14, freq="D"):
        rows.append(f"{day.strftime('%Y-%m-%d')},No Promotion,1000,1,500,100")
    csv_content = '"","Date","Sales_name","gmv_new","units","product_mrp","discount"\n'
    for i, row in enumerate(rows):
        csv_content += f"{i},{row}\n"
    (tmp_path / "firstfile.csv").write_text(csv_content)

    result = aggregate_weekly_gmv(tmp_path)
    assert "Date" in result.columns
    assert "total_gmv" in result.columns
    assert "total_Discount" in result.columns
    # 14 days starting Monday Jul 6 → 2 complete weeks
    assert len(result) == 2
    assert result["total_gmv"].sum() == 14000


def test_aggregate_weekly_gmv_file_not_found(tmp_path):
    """Test that aggregate_weekly_gmv raises when file is missing."""
    with pytest.raises(FileNotFoundError):
        aggregate_weekly_gmv(tmp_path)


def test_distribute_media_to_weekly(tmp_path):
    """Test pro-rata distribution of monthly media spend to weeks."""
    media_csv = (
        "Year,Month,Total Investment,TV,Digital,Sponsorship,"
        "Content Marketing,Online marketing, Affiliates,SEM,Radio,Other\n"
        "2015,7,10,2,1,3,0,1,0.5,2.5,,\n"
    )
    (tmp_path / "MediaInvestment.csv").write_text(media_csv)

    # 4 Mondays in July 2015: Jul 6, 13, 20, 27
    weekly_dates = pd.Series(
        pd.to_datetime(["2015-07-06", "2015-07-13", "2015-07-20", "2015-07-27"])
    )

    result = distribute_media_to_weekly(tmp_path, weekly_dates)
    assert len(result) == 4
    assert "TV" in result.columns
    # TV monthly = 2 * 1e7 = 2e7, split across 4 weeks = 5e6 each
    np.testing.assert_allclose(result["TV"].values, 5e6, rtol=1e-6)


def test_distribute_media_file_not_found(tmp_path):
    """Test that distribute_media_to_weekly raises when file is missing."""
    weekly_dates = pd.Series(pd.to_datetime(["2015-07-06"]))
    with pytest.raises(FileNotFoundError):
        distribute_media_to_weekly(tmp_path, weekly_dates)


def test_distribute_nps_to_weekly(tmp_path):
    """Test NPS assignment from monthly to weekly."""
    nps_csv = "Date,NPS\n7/1/2015,54.6\n8/1/2015,60\n"
    (tmp_path / "MonthlyNPSscore.csv").write_text(nps_csv)

    weekly_dates = pd.Series(pd.to_datetime(["2015-07-06", "2015-07-13", "2015-08-03"]))
    result = distribute_nps_to_weekly(tmp_path, weekly_dates)
    assert result.loc[0, "NPS"] == 54.6
    assert result.loc[1, "NPS"] == 54.6
    assert result.loc[2, "NPS"] == 60


def test_distribute_nps_file_not_found(tmp_path):
    """Test that distribute_nps_to_weekly raises when file is missing."""
    weekly_dates = pd.Series(pd.to_datetime(["2015-07-06"]))
    with pytest.raises(FileNotFoundError):
        distribute_nps_to_weekly(tmp_path, weekly_dates)


def test_compute_weekly_sale_days(tmp_path):
    """Test weekly sale day counting from SpecialSale.csv."""
    # Two events in the same week (Mon Jul 13), one in another week (Mon Jul 20)
    special_csv = (
        "Date,Sales Name\n7/18/2015,Sale A\n7/19/2015,Sale A\n7/20/2015,Sale B\n"
    )
    (tmp_path / "SpecialSale.csv").write_text(special_csv)

    # Jul 13 week contains Jul 18-19 (Sat-Sun), Jul 20 week contains Jul 20 (Mon)
    weekly_dates = pd.Series(
        pd.to_datetime(["2015-07-06", "2015-07-13", "2015-07-20", "2015-07-27"])
    )
    result = compute_weekly_sale_days(tmp_path, weekly_dates)
    assert len(result) == 4
    assert result.loc[0, "sale_days"] == 0  # Jul 6 week — no events
    assert result.loc[1, "sale_days"] == 2  # Jul 13 week — Jul 18, 19
    assert result.loc[2, "sale_days"] == 1  # Jul 20 week — Jul 20
    assert result.loc[3, "sale_days"] == 0  # Jul 27 week — no events


def test_aggregate_channel_groups_default():
    """Test that aggregate_channel_groups sums raw channels into groups."""
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2015-07-06", periods=3, freq="W-MON"),
            "TV": [100, 200, 300],
            "Digital": [10, 20, 30],
            "SEM": [5, 10, 15],
            "Content.Marketing": [1, 2, 3],
            "Online.marketing": [50, 60, 70],
            "Affiliates": [25, 30, 35],
            "Sponsorship": [400, 500, 600],
        }
    )
    result = aggregate_channel_groups(df)
    assert list(result["TV"]) == [100, 200, 300]
    assert list(result["Sponsorship"]) == [400, 500, 600]
    # Digital = Digital + SEM + Content.Marketing
    assert list(result["Digital"]) == [16, 32, 48]
    # Online = Online.marketing + Affiliates
    assert list(result["Online"]) == [75, 90, 105]
    # Raw columns should be dropped
    for col in ["SEM", "Content.Marketing", "Online.marketing", "Affiliates"]:
        assert col not in result.columns


def test_aggregate_channel_groups_missing_column():
    """Test that aggregate_channel_groups raises on missing source columns."""
    df = pd.DataFrame({"TV": [1], "Sponsorship": [2]})
    with pytest.raises(ValueError, match="Missing source channel columns"):
        aggregate_channel_groups(df)


def test_aggregate_channel_groups_custom_groups():
    """Test aggregate_channel_groups with custom group mapping."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = aggregate_channel_groups(df, groups={"ab": ["a", "b"], "c": ["c"]})
    assert list(result["ab"]) == [4, 6]
    assert list(result["c"]) == [5, 6]
    assert "a" not in result.columns
    assert "b" not in result.columns


def test_validate_mmm_weekly_data_passes(mmm_weekly_sample_dataframe):
    """Test validation passes for properly structured weekly data."""
    validate_mmm_weekly_data(mmm_weekly_sample_dataframe)


def test_validate_mmm_weekly_data_missing_column(mmm_weekly_sample_dataframe):
    """Test validation fails when a required column is missing."""
    df = mmm_weekly_sample_dataframe.drop(columns=["NPS"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_mmm_weekly_data(df)


def test_validate_mmm_weekly_data_wrong_row_count(mmm_weekly_sample_dataframe):
    """Test validation fails when row count is outside 50-53."""
    df = mmm_weekly_sample_dataframe.head(10)
    with pytest.raises(ValueError, match="Expected 50-53"):
        validate_mmm_weekly_data(df)
