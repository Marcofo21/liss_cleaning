"""Tests for matching_probabilities cleaner functions."""

import pandas as pd
import pytest

from liss_cleaning.make_final_datasets.cleaners.matching_probabilities import (
    _check_answered_all_questions,
    _check_eligible,
    _get_interval,
)


class TestCheckAnsweredAllQuestions:
    def test_returns_true_when_all_answered(self):
        df = pd.DataFrame({
            "mp_e0": [(0, 0.1)],
            "mp_e1": [(0.1, 0.2)],
            "mp_e2": [(0.2, 0.3)],
            "mp_e3": [(0.3, 0.4)],
            "mp_e1c": [(0.4, 0.5)],
            "mp_e2c": [(0.5, 0.6)],
            "mp_e3c": [(0.6, 0.7)],
        })
        assert _check_answered_all_questions(df)

    def test_returns_false_when_one_missing(self):
        df = pd.DataFrame({
            "mp_e0": [pd.NA],
            "mp_e1": [(0.1, 0.2)],
            "mp_e2": [(0.2, 0.3)],
            "mp_e3": [(0.3, 0.4)],
            "mp_e1c": [(0.4, 0.5)],
            "mp_e2c": [(0.5, 0.6)],
            "mp_e3c": [(0.6, 0.7)],
        })
        assert not _check_answered_all_questions(df)

    def test_returns_false_when_all_missing(self):
        df = pd.DataFrame({
            "mp_e0": [pd.NA],
            "mp_e1": [pd.NA],
            "mp_e2": [pd.NA],
            "mp_e3": [pd.NA],
            "mp_e1c": [pd.NA],
            "mp_e2c": [pd.NA],
            "mp_e3c": [pd.NA],
        })
        assert not _check_answered_all_questions(df)


class TestCheckEligible:
    def test_returns_true_with_two_waves(self):
        df = pd.DataFrame({
            "personal_id": [1, 1],
            "wave": [1, 2],
            "mp_e0": [(0, 0.1), (0, 0.1)],
        })
        assert _check_eligible(df)

    def test_returns_false_with_one_wave(self):
        df = pd.DataFrame({
            "personal_id": [1],
            "wave": [1],
            "mp_e0": [(0, 0.1)],
        })
        assert not _check_eligible(df)

    def test_returns_true_with_many_waves(self):
        df = pd.DataFrame({
            "personal_id": [1, 1, 1, 1],
            "wave": [1, 2, 3, 4],
            "mp_e0": [(0, 0.1)] * 4,
        })
        assert _check_eligible(df)


class TestGetInterval:
    @pytest.fixture
    def make_choice_row(self):
        """Factory to create a row with specified choices."""
        def _make_row(option, choices):
            """Create a series with choice columns for a given option.

            Args:
                option: The ambiguous option (e.g., 'e0')
                choices: Dict mapping probability to choice ('AEX' or 'Lottery')
            """
            data = {}
            for prob in ["50", "90", "10", "95", "70", "30", "5", "99", "80", "60",
                         "40", "20", "1"]:
                col = f"choice_aex_{option}_vs_{prob}"
                data[col] = choices.get(prob, pd.NA)
            return pd.Series(data)
        return _make_row

    def test_returns_na_when_all_na(self, make_choice_row):
        row = make_choice_row("e0", {})
        result = _get_interval(row, "e0")
        assert pd.isna(result)

    def test_high_confidence_aex_returns_99_100(self, make_choice_row):
        row = make_choice_row("e0", {
            "50": "AEX",
            "90": "AEX",
            "95": "AEX",
            "99": "AEX",
        })
        result = _get_interval(row, "e0")
        assert result == (0.99, 1)

    def test_low_confidence_lottery_returns_0_001(self, make_choice_row):
        row = make_choice_row("e0", {
            "50": "Lottery",
            "10": "Lottery",
            "5": "Lottery",
            "1": "Lottery",
        })
        result = _get_interval(row, "e0")
        assert result == (0, 0.01)

    def test_middle_range_returns_correct_interval(self, make_choice_row):
        row = make_choice_row("e0", {
            "50": "AEX",
            "90": "Lottery",
            "70": "AEX",
            "80": "Lottery",
        })
        result = _get_interval(row, "e0")
        assert result == (0.7, 0.8)
