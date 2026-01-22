"""Tests for general_cleaners helper functions."""

import pandas as pd
import pytest

from liss_cleaning.helper_modules.general_cleaners import (
    _apply_lowest_int_dtype,
    _find_lowest_int_dtype,
    _handle_inconsistent_column_code_in_raw,
    _handle_missing_column,
    _replace_missing_floats,
    _replace_values,
)


class TestReplaceValues:
    def test_replaces_value_in_dict(self):
        result = _replace_values("ja", {"ja": "Yes", "nee": "No"})
        assert result == "Yes"

    def test_returns_original_if_not_in_dict(self):
        result = _replace_values("unknown", {"ja": "Yes", "nee": "No"})
        assert result == "unknown"

    def test_handles_none_value(self):
        result = _replace_values(None, {"ja": "Yes"})
        assert result is None


class TestHandleMissingColumn:
    def test_returns_series_when_column_exists(self):
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        result = _handle_missing_column(df, "col_a")
        assert result["is_missing"] is False

    def test_returns_existing_series_values(self):
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        result = _handle_missing_column(df, "col_a")
        pd.testing.assert_series_equal(result["series"], df["col_a"])

    def test_returns_na_series_when_column_missing(self):
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        result = _handle_missing_column(df, "col_b")
        assert result["is_missing"] is True

    def test_na_series_has_correct_length(self):
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        result = _handle_missing_column(df, "col_b")
        assert len(result["series"]) == 3

    def test_na_series_contains_only_na(self):
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        result = _handle_missing_column(df, "col_b")
        assert result["series"].isna().all()


class TestReplaceMissingFloats:
    def test_replaces_single_nan_value(self):
        series = pd.Series([1.0, 9999999999, 3.0])
        result = _replace_missing_floats(series, [9999999999])
        assert pd.isna(result.iloc[1])

    def test_replaces_multiple_nan_values(self):
        series = pd.Series([1.0, 9999999999, 9999999998, 3.0])
        result = _replace_missing_floats(series, [9999999999, 9999999998])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_preserves_valid_values(self):
        series = pd.Series([1.0, 9999999999, 3.0])
        result = _replace_missing_floats(series, [9999999999])
        assert result.iloc[0] == 1.0
        assert result.iloc[2] == 3.0


class TestFindLowestIntDtype:
    def test_small_positive_uses_uint8(self):
        series = pd.Series([0, 100, 255])
        result = _find_lowest_int_dtype(series)
        assert result == "uint8[pyarrow]"

    def test_larger_positive_uses_uint16(self):
        series = pd.Series([0, 1000, 65535])
        result = _find_lowest_int_dtype(series)
        assert result == "uint16[pyarrow]"

    def test_negative_values_use_signed_int(self):
        series = pd.Series([-10, 0, 10])
        result = _find_lowest_int_dtype(series)
        assert result == "int8[pyarrow]"

    def test_large_negative_uses_int16(self):
        series = pd.Series([-1000, 0, 1000])
        result = _find_lowest_int_dtype(series)
        assert result == "int16[pyarrow]"


class TestApplyLowestIntDtype:
    def test_converts_to_pyarrow_dtype(self):
        series = pd.Series([1, 2, 3])
        result = _apply_lowest_int_dtype(series)
        assert "pyarrow" in str(result.dtype)

    def test_preserves_values(self):
        series = pd.Series([1, 2, 3])
        result = _apply_lowest_int_dtype(series)
        assert list(result) == [1, 2, 3]


class TestHandleInconsistentColumnCodeInRaw:
    def test_returns_post_change_code_after_switch_year(self):
        result = _handle_inconsistent_column_code_in_raw(
            pre_change_code=112,
            post_change_code=368,
            year_switch=2014,
            year_current_df=2020,
        )
        assert result == "368"

    def test_returns_pre_change_code_before_switch_year(self):
        result = _handle_inconsistent_column_code_in_raw(
            pre_change_code=112,
            post_change_code=368,
            year_switch=2014,
            year_current_df=2010,
        )
        assert result == "112"

    def test_returns_post_change_code_at_switch_year(self):
        result = _handle_inconsistent_column_code_in_raw(
            pre_change_code=112,
            post_change_code=368,
            year_switch=2014,
            year_current_df=2014,
        )
        assert result == "368"
