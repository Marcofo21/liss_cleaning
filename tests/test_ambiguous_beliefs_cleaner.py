"""Tests for ambiguous_beliefs_cleaner helper functions."""

import pandas as pd
import pytest

from liss_cleaning.raw_datasets_cleaning.cleaners.ambiguous_beliefs_cleaner import (
    _clean_aex_choice,
    _clean_attention_check_ja_fail,
    _clean_attention_check_ja_pass,
    _extract_wave_identifier,
    _parse_date_str,
    _parse_time_str,
)


class TestExtractWaveIdentifier:
    def test_extracts_wave_from_typical_filename(self):
        result = _extract_wave_identifier("survey_ab_3_2019.dta")
        assert result == 3

    def test_extracts_single_digit_wave(self):
        result = _extract_wave_identifier("survey_ab_1_2018.dta")
        assert result == 1

    def test_extracts_double_digit_wave(self):
        result = _extract_wave_identifier("survey_ab_12_2022.dta")
        assert result == 12


class TestCleanAttentionCheckJaPass:
    def test_ja_maps_to_passed(self):
        result = _clean_attention_check_ja_pass(pd.Series(["ja"]))
        assert result.iloc[0] == "Passed"

    def test_nee_maps_to_failed(self):
        result = _clean_attention_check_ja_pass(pd.Series(["nee"]))
        assert result.iloc[0] == "Failed"

    def test_returns_categorical_dtype(self):
        result = _clean_attention_check_ja_pass(pd.Series(["ja", "nee"]))
        assert isinstance(result.dtype, pd.CategoricalDtype)

    def test_categories_are_passed_failed(self):
        result = _clean_attention_check_ja_pass(pd.Series(["ja"]))
        assert list(result.cat.categories) == ["Passed", "Failed"]


class TestCleanAttentionCheckJaFail:
    def test_ja_maps_to_failed(self):
        result = _clean_attention_check_ja_fail(pd.Series(["ja"]))
        assert result.iloc[0] == "Failed"

    def test_nee_maps_to_passed(self):
        result = _clean_attention_check_ja_fail(pd.Series(["nee"]))
        assert result.iloc[0] == "Passed"


class TestCleanAexChoice:
    def test_optie_1_maps_to_aex(self):
        result = _clean_aex_choice(pd.Series(["optie 1"]))
        assert result.iloc[0] == "AEX"

    def test_optie_2_maps_to_lottery(self):
        result = _clean_aex_choice(pd.Series(["optie 2"]))
        assert result.iloc[0] == "Lottery"

    def test_returns_categorical_dtype(self):
        result = _clean_aex_choice(pd.Series(["optie 1", "optie 2"]))
        assert isinstance(result.dtype, pd.CategoricalDtype)

    def test_categories_are_aex_lottery(self):
        result = _clean_aex_choice(pd.Series(["optie 1"]))
        assert list(result.cat.categories) == ["AEX", "Lottery"]


class TestParseTimeStr:
    def test_parses_valid_time(self):
        result = _parse_time_str("14:30:00")
        assert result.hour == 14
        assert result.minute == 30

    def test_empty_string_returns_na(self):
        result = _parse_time_str("")
        assert pd.isna(result)

    def test_space_returns_na(self):
        result = _parse_time_str(" ")
        assert pd.isna(result)


class TestParseDateStr:
    def test_parses_valid_date(self):
        result = _parse_date_str("15-03-2019")
        assert result.day == 15
        assert result.month == 3
        assert result.year == 2019

    def test_empty_string_returns_na(self):
        result = _parse_date_str("")
        assert pd.isna(result)

    def test_space_returns_na(self):
        result = _parse_date_str(" ")
        assert pd.isna(result)
