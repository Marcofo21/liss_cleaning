import pandas as pd

from liss_cleaning.config import SRC_DATA
from liss_cleaning.helper_modules.general_cleaners import (
    _apply_lowest_int_dtype,
    _replace_values,
)

pd.set_option("future.no_silent_downcasting", True)

dependencies_time_index = {
    SRC_DATA / "xxx-ambiguous-beliefs" / "wave-1" / "L_gaudecker2018_1_6p.dta": 1,
    SRC_DATA / "xxx-ambiguous-beliefs" / "wave-2" / "L_gaudecker2018_2_6p.dta": 2,
    SRC_DATA / "xxx-ambiguous-beliefs" / "wave-3" / "L_gaudecker2019_3_6p.dta": 3,
    SRC_DATA / "xxx-ambiguous-beliefs" / "wave-4" / "L_gaudecker2019_4_6p.dta": 4,
    SRC_DATA / "xxx-ambiguous-beliefs" / "wave-5" / "L_gaudecker2020_5_6p.dta": 5,
    SRC_DATA / "xxx-ambiguous-beliefs" / "wave-6" / "L_gaudecker2020_6_6p.dta": 6,
    SRC_DATA / "xxx-ambiguous-beliefs" / "wave-7" / "L_gaudecker2021_7_6p.dta": 7,
    "index_name": "wave",
}


def clean_dataset(
    raw,
    source_file_name,
) -> pd.DataFrame:
    """Clean the ambiguous beliefs dataset."""
    cleaned_data = pd.DataFrame(index=raw.index)
    wave_identifier = dependencies_time_index[source_file_name]
    cleaned_data["wave"] = wave_identifier
    wave_to_year = {
        1: 2018,
        2: 2018,
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
        1: "e0",
        2: "e1",
        3: "e2",
        4: "e3",
        5: "e1c",
        6: "e2c",
        7: "e3c",
    }

    options_2 = {
        1: "50",
        2: "90",
        3: "10",
        4: "95",
        5: "70",
        6: "30",
        7: "5",
        8: "99",
        9: "80",
        10: "60",
        11: "40",
        12: "20",
        13: "1",
    }

    for option_1_key, option_1_value in options_1.items():
        for option_2_key, option_2_value in options_2.items():
            cleaned_data[f"choice_aex_{option_1_value}_vs_{option_2_value}"] = (
                pd.Categorical(
                    raw[f"keuze_{option_1_key}_{option_2_key}"].apply(
                        lambda x: _replace_values(
                            x,
                            {"optie 1": "AEX", "optie 2": "Lottery"},
                        ),
                    ),
                    categories=["AEX", "Lottery"],
                )
            )

    cleaned_data["end_time"] = raw["TijdE"].apply(_clean_time_str)
    cleaned_data["start_time"] = raw["TijdB"].apply(_clean_time_str)
    cleaned_data["data_completion"] = raw["DatumE"].apply(_clean_date_str)
    return cleaned_data


def _clean_time_str(time_str):
    """Adjust time string to have only hours, minutes and seconds."""
    if time_str in (" ", ""):
        return pd.NA
    return pd.to_datetime(time_str, format="%H:%M:%S")


def _clean_date_str(date_str):
    """Adjust date string to have only year, month and day."""
    if date_str in (" ", ""):
        return pd.NA
    return pd.to_datetime(date_str, format="%d-%m-%Y")
