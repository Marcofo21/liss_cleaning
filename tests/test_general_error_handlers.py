"""Tests for general_error_handlers module."""

from pathlib import Path

import pandas as pd
import pytest

from liss_cleaning.helper_modules.general_error_handlers import (
    _check_file_exists,
    _check_object_type,
    _check_series_dtype,
    _check_variables_exist,
)


class TestCheckVariablesExist:
    def test_passes_when_all_variables_exist(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        _check_variables_exist(df, ["a", "b"])

    def test_raises_when_variable_missing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Variable c not in data columns"):
            _check_variables_exist(df, ["a", "c"])

    def test_passes_with_empty_variable_list(self):
        df = pd.DataFrame({"a": [1]})
        _check_variables_exist(df, [])


class TestCheckObjectType:
    def test_passes_for_correct_type(self):
        _check_object_type([1, 2, 3], list)

    def test_raises_for_wrong_type(self):
        with pytest.raises(TypeError, match="Expected .* to be of type"):
            _check_object_type("not a list", list)

    def test_passes_for_dataframe(self):
        df = pd.DataFrame({"a": [1]})
        _check_object_type(df, pd.DataFrame)


class TestCheckFileExists:
    def test_passes_for_existing_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        _check_file_exists(test_file)

    def test_raises_for_nonexistent_file(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _check_file_exists(Path("/nonexistent/path/file.txt"))

    def test_accepts_string_path(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        _check_file_exists(str(test_file))


class TestCheckSeriesDtype:
    def test_passes_for_correct_dtype(self):
        series = pd.Series([1, 2, 3], dtype="int64")
        _check_series_dtype(series, "int64")

    def test_raises_for_wrong_dtype(self):
        series = pd.Series([1, 2, 3], dtype="int64")
        with pytest.raises(TypeError, match="Expected series to have dtype"):
            _check_series_dtype(series, "float64")
