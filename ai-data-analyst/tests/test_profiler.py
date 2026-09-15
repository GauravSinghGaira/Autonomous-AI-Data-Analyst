import pandas as pd
import pytest

from app.data_processing.profiler import profile_dataset, query_dataframe


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "id": range(1, 21),
        "age": [22, 25, 29, 31, 40, 22, 25, 29, 31, 40, 22, 25, 29, 31, 40, 22, 25, 29, 31, None],
        "salary": [50000, 60000, 70000, 80000, 90000] * 4,
        "department": ["eng", "sales", "eng", "hr", "sales"] * 4,
    })


def test_profile_basic_shape(sample_df):
    profile = profile_dataset(sample_df)
    assert profile["n_rows"] == 20
    assert profile["n_cols"] == 4


def test_profile_missing_values(sample_df):
    profile = profile_dataset(sample_df)
    age_col = next(c for c in profile["columns"] if c["name"] == "age")
    assert age_col["missing_count"] == 1


def test_profile_duplicate_detection():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    profile = profile_dataset(df)
    assert profile["duplicate_rows"] == 1


def test_query_dataframe_mean(sample_df):
    result = query_dataframe(sample_df, column="salary", agg="mean")
    assert result["result"] == pytest.approx(70000.0)


def test_query_dataframe_groupby(sample_df):
    result = query_dataframe(sample_df, column="salary", agg="mean", groupby="department")
    assert "eng" in result["result"]


def test_query_dataframe_unknown_column(sample_df):
    result = query_dataframe(sample_df, column="not_a_column", agg="mean")
    assert "error" in result
