"""Cleaner for the ambiguous beliefs survey dataset."""

import pandas as pd

from liss_cleaning.helper_modules.general_cleaners import _apply_lowest_int_dtype

WAVE_TO_YEAR = {
    1: 2018,
    2: 2018,
    3: 2019,
    4: 2019,
    5: 2020,
    6: 2020,
    7: 2021,
}

AMBIGUOUS_OPTIONS = {
    1: "e0",
    2: "e1",
    3: "e2",
    4: "e3",
    5: "e1c",
    6: "e2c",
    7: "e3c",
}

LOTTERY_PROBABILITIES = {
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


def clean_dataset(raw, source_file_name):
    """Clean the ambiguous beliefs dataset.

    Args:
        raw: Raw DataFrame from .dta file.
        source_file_name: Name of the source file (used to extract wave identifier).

    Returns:
        Cleaned DataFrame with standardized column names and types.

    """
    df = pd.DataFrame(index=raw.index)
    wave_identifier = _extract_wave_identifier(source_file_name)

    df["wave"] = wave_identifier
    df["year"] = WAVE_TO_YEAR[wave_identifier]
    df["personal_id"] = _apply_lowest_int_dtype(raw["nomem_encr"])
    df["attention_check_1"] = _clean_attention_check_ja_pass(raw["check_aex"])
    df["attention_check_2_rad"] = _clean_attention_check_ja_fail(raw["check_rad"])
    df["attention_check_3_rad"] = _clean_attention_check_ja_fail(raw["check_rad2"])
    df["attention_check_4_fb"] = _clean_attention_check_ja_pass(raw["check_aex2"])

    for opt1_key, opt1_val in AMBIGUOUS_OPTIONS.items():
        for opt2_key, opt2_val in LOTTERY_PROBABILITIES.items():
            col_name = f"choice_aex_{opt1_val}_vs_{opt2_val}"
            df[col_name] = _clean_aex_choice(raw[f"keuze_{opt1_key}_{opt2_key}"])

    df["end_time"] = _clean_time(raw["TijdE"])
    df["start_time"] = _clean_time(raw["TijdB"])
    df["data_completion"] = _clean_date(raw["DatumE"])

    return df


def _extract_wave_identifier(source_file_name):
    """Extract wave number from source file name."""
    return int(source_file_name.split("_")[2])


def _clean_attention_check_ja_pass(series):
    """Clean attention check where 'ja' means passed."""
    mapping = {"ja": "Passed", "nee": "Failed"}
    return pd.Categorical(
        series.map(mapping),
        categories=["Passed", "Failed"],
    )


def _clean_attention_check_ja_fail(series):
    """Clean attention check where 'ja' means failed."""
    mapping = {"ja": "Failed", "nee": "Passed"}
    return pd.Categorical(
        series.map(mapping),
        categories=["Passed", "Failed"],
    )


def _clean_aex_choice(series):
    """Clean AEX vs lottery choice column."""
    mapping = {"optie 1": "AEX", "optie 2": "Lottery"}
    return pd.Categorical(
        series.map(mapping),
        categories=["AEX", "Lottery"],
    )


def _clean_time(series):
    """Parse time string to datetime, treating blanks as NA."""
    return series.apply(_parse_time_str)


def _parse_time_str(time_str):
    """Convert time string to datetime or NA if blank."""
    if time_str in (" ", ""):
        return pd.NA
    return pd.to_datetime(time_str, format="%H:%M:%S")


def _clean_date(series):
    """Parse date string to datetime, treating blanks as NA."""
    return series.apply(_parse_date_str)


def _parse_date_str(date_str):
    """Convert date string to datetime or NA if blank."""
    if date_str in (" ", ""):
        return pd.NA
    return pd.to_datetime(date_str, format="%d-%m-%Y")
