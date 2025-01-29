import pandas as pd

from liss_cleaning.config import SRC_DATA
from liss_cleaning.helper_modules.general_cleaners import (
    _apply_lowest_int_dtype,
    _replace_values,
)

pd.set_option("future.no_silent_downcasting", True)

survey_time_index = {
    f"{SRC_DATA}/xxx-ambiguous-beliefs/wave-3/L_gaudecker2019_3_6p.dta": 3,
    f"{SRC_DATA}/xxx-ambiguous-beliefs/wave-4/L_gaudecker2019_4_6p.dta": 4,
    f"{SRC_DATA}/xxx-ambiguous-beliefs/wave-5/L_gaudecker2020_5_6p.dta": 5,
    f"{SRC_DATA}/xxx-ambiguous-beliefs/wave-6/L_gaudecker2020_6_6p.dta": 6,
    f"{SRC_DATA}/xxx-ambiguous-beliefs/wave-7/L_gaudecker2021_7_6p.dta": 7,
    "index_name": "wave",
}


def clean_ambiguous_beliefs(
    raw,
    source_file_name,
) -> pd.DataFrame:
    """Clean the ambiguous beliefs dataset."""
    cleaned_data = pd.DataFrame(index=raw.index)
    wave_identifier = survey_time_index[source_file_name]
    cleaned_data["wave"] = wave_identifier
    wave_to_year = {
        3: 2019,
        4: 2019,
        5: 2020,
        6: 2020,
        7: 2021,
    }
    cleaned_data["year"] = wave_to_year[wave_identifier]
    cleaned_data["personal_id"] = _apply_lowest_int_dtype(raw["nomem_encr"])
    cleaned_data["attention_check_1"] = pd.Categorical(
        raw["check_aex"].apply(
            lambda x: _replace_values(x, {"ja": "Passed", "nee": "Failed"}),
        ),
        categories=["Passed", "Failed"],
    )
    cleaned_data["attention_check_2_rad"] = pd.Categorical(
        raw["check_rad"].apply(
            lambda x: _replace_values(x, {"ja": "Failed", "nee": "Passed"}),
        ),
        categories=["Passed", "Failed"],
    )
    cleaned_data["attention_check_3_rad"] = pd.Categorical(
        raw["check_rad2"].apply(
            lambda x: _replace_values(x, {"ja": "Failed", "nee": "Passed"}),
        ),
        categories=["Passed", "Failed"],
    )
    cleaned_data["attention_check_4_fb"] = pd.Categorical(
        raw["check_aex2"].apply(
            lambda x: _replace_values(x, {"ja": "Passed", "nee": "Failed"}),
        ),
        categories=["Passed", "Failed"],
    )

    options_1 = {
        1: ">1000",
        2: ">1100",
        3: "<950",
        4: ">950_<1100",
        5: "<=1100",
        6: ">=950",
        7: "<950_>1100",
    }

    options_2 = {
        1: "50",
        2: "90",
        3: "95",
        4: "99",
        5: "70",
        6: "80",
        7: "60",
        8: "10",
        9: "30",
        10: "40",
        11: "20",
        12: "5",
        13: "1",
    }

    for option_1_key, option_1_value in options_1.items():
        for option_2_key, option_2_value in options_2.items():
            cleaned_data[f"choice_aex_{option_1_value}_vs_{option_2_value}"] = (
                pd.Categorical(
                    raw[f"keuze_{option_1_key}_{option_2_key}"].apply(
                        lambda x: _replace_values(
                            x,
                            {"optie 1": "Ambiguous bet", "optie 2": "Unambiguous bet"},
                        ),
                    ),
                    categories=["Ambiguous bet", "Unambiguous bet"],
                )
            )

    return cleaned_data
