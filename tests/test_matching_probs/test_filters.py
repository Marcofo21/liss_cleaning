import pandas as pd
import pytest

# Import the functions under test
from liss_cleaning.data_cleaning.make_normalized_datasets.specific_cleaners.matching_probabilities_cleaner import (  # noqa: E501
    _check_answered_all_questions,
)


@pytest.fixture
def individuals():
    return {
        "answered_all_participated_waves": pd.DataFrame(
            {
                "personal_id": [1],
                "wave": [1],
                "mp_e0": [(0, 0.1)],
                "mp_e1": [(0.1, 0.2)],
                "mp_e2": [(0.2, 0.3)],
                "mp_e3": [(0.3, 0.4)],
                "mp_e1c": [(0.4, 0.5)],
                "mp_e2c": [(0.5, 0.6)],
                "mp_e3c": [(0.6, 0.7)],
            }
        ),
        "not_answered_all_participated_waves": pd.DataFrame(
            {
                "personal_id": [1],
                "wave": [1],
                "mp_e0": [pd.NA],
                "mp_e1": [(0.1, 0.2)],
                "mp_e2": [(0.2, 0.3)],
                "mp_e3": [(0.3, 0.4)],
                "mp_e1c": [(0.4, 0.5)],
                "mp_e2c": [(0.5, 0.6)],
                "mp_e3c": [(0.6, 0.7)],
            }
        ),
    }


def test_check_answered_all_participated_waves(individuals):
    df_true = individuals["answered_all_participated_waves"]
    df_false = individuals["not_answered_all_participated_waves"]
    assert _check_answered_all_questions(df_true)
    assert not _check_answered_all_questions(df_false)
