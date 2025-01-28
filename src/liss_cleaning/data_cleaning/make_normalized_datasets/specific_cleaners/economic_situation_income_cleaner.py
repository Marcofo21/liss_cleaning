import numpy as np
import pandas as pd
from liss_data_cleaning.config import SRC_DATA
from liss_data_cleaning.helper_modules.general_cleaners import (
    _apply_lowest_float_dtype,
    _apply_lowest_int_dtype,
    _handle_inconsistent_column_code_in_raw,
    _handle_missing_column,
    _replace_missing_floats,
    _replace_mixed_categoricals_floats,
    _replace_rename_categorical_column,
)

pd.set_option("future.no_silent_downcasting", True)

survey_time_index = {
    f"{SRC_DATA}/010-economic-situation-income/wave-1/ci08a_1.0p_EN.dta": 2008,
    f"{SRC_DATA}/010-economic-situation-income/wave-2/ci09b_EN_1.1p.dta": 2009,
    f"{SRC_DATA}/010-economic-situation-income/wave-3/ci10c_EN_1.0p.dta": 2010,
    f"{SRC_DATA}/010-economic-situation-income/wave-4/ci11d_EN_1.0p.dta": 2011,
    f"{SRC_DATA}/010-economic-situation-income/wave-5/ci12e_1.0p_EN.dta": 2012,
    f"{SRC_DATA}/010-economic-situation-income/wave-6/ci13f_1.1p_EN.dta": 2013,
    f"{SRC_DATA}/010-economic-situation-income/wave-7/ci14g_1.0p_EN.dta": 2014,
    f"{SRC_DATA}/010-economic-situation-income/wave-8/ci15h_EN_1.0p.dta": 2015,
    f"{SRC_DATA}/010-economic-situation-income/wave-9/ci16i_EN_1.0p.dta": 2016,
    f"{SRC_DATA}/010-economic-situation-income/wave-10/ci17j_EN_1.0p.dta": 2017,
    f"{SRC_DATA}/010-economic-situation-income/wave-11/ci18k_EN_1.0p.dta": 2018,
    f"{SRC_DATA}/010-economic-situation-income/wave-12/ci19l_EN_1.0p.dta": 2019,
    f"{SRC_DATA}/010-economic-situation-income/wave-13/ci20m_EN_1.0p.dta": 2020,
    f"{SRC_DATA}/010-economic-situation-income/wave-14/ci21n_EN_1.0p.dta": 2021,
    "index_name": "year",
}


def clean_economic_situation_income(raw, source_file_name) -> pd.DataFrame:
    """Clean the economic situation income data from the LISS panel.

    Args:
        raw (pd.DataFrame): The raw data.
        source_file_name (str): The name of the source file.

    Returns:
        pd.DataFrame: The cleaned data.
    """
    cleaned = pd.DataFrame(index=raw.index)
    column_time_identifier = source_file_name.split("/")[-1].split("_")[0][2:5]

    cleaned[survey_time_index["index_name"]] = survey_time_index[source_file_name]
    cleaned["personal_id"] = _apply_lowest_int_dtype(raw["nomem_encr"])
    cleaned["age"] = _apply_lowest_int_dtype(raw[f"ci{column_time_identifier}002"])
    cleaned["alimony_children_amt"] = _replace_mixed_categoricals_floats(
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
        series=raw[f"ci{column_time_identifier}208"],
    )
    cleaned["alimony_partner_amt"] = _replace_mixed_categoricals_floats(
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
        series=raw[f"ci{column_time_identifier}206"],
    )

    renaming_dict_appliances_columns = {
        "yes": "Yes",
        "no (not affordable)": "No",
        "no (not necessary)": "No",
        "no (other reason)": "No",
        "no (don't need it)": "No",
        "no (can't afford)": "No",
        "don't know": pd.NA,
        "nan": pd.NA,
        "don\x92\t know": pd.NA,
        "don\x92t know": pd.NA,
    }
    appliances_columns_to_code = {
        "camcorder": 279,
        "car": 348,
        "cd_dvd_writer": 276,
        "cd_player": 273,
        "computer": 282,
        "deep_fryer": 291,
        "digital_camera": 284,
        "digital_tv": 270,
        "dishwasher": 288,
        "dvd_player": 274,
        "dvd_recorder": 275,
        "fixed_line_phone": 266,
        "freezer": 289,
        "games_console": 285,
        "gps": 286,
        "home_cinema": 272,
        "microwave": 290,
        "mp3_player": 277,
        "mp4_player": 278,
        "pda_with_inet": 281,
        "pda_without_inet": 280,
        "phone": 349,
        "phone_w_inet": 268,
        "phone_wo_inet": 267,
        "printer": 283,
        "satellite_dish": 271,
        "widescreen_tv": 269,
        "wash_dryer": 287,
    }
    for appliance, column_code in appliances_columns_to_code.items():
        handle_missing_dict = _handle_missing_column(
            raw,
            f"ci{column_time_identifier}{column_code}",
        )
        series = handle_missing_dict["series"]
        is_missing = handle_missing_dict["is_missing"]
        cleaned[f"appliances_has_{appliance}"] = _replace_rename_categorical_column(
            series,
            renaming_dict_appliances_columns,
            is_missing=is_missing,
        )

    cleaned["appliances_reason_nophone"] = _replace_rename_categorical_column(
        **_handle_missing_column(raw, f"ci{column_time_identifier}265"),
        renaming_dict={
            np.nan: pd.NA,
            99: pd.NA,
            98: pd.NA,
            "don't need it": "Don't need it",
            "can't afford it": "Can't afford",
        },
    )

    arrear_amt_columns_to_code = {
        "other_bills": 300,
        "rent_mortgage": _handle_inconsistent_column_code_in_raw(
            298,
            381,
            2019,
            survey_time_index[source_file_name],
        ),
        "utilities": 299,
    }

    for arrear, column_code in arrear_amt_columns_to_code.items():
        cleaned[f"arrears_total_amount_{arrear}"] = _replace_mixed_categoricals_floats(
            float_nan_values=[9999999999, 9999999998],
            categories_nan_entries=["I don't know", "I prefer not to say"],
            series=raw[f"ci{column_time_identifier}{column_code}"],
        )

    cleaned["arrears_longest_duration_m_rent"] = _replace_mixed_categoricals_floats(
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
        series=raw[f"ci{column_time_identifier}{column_code}"],
    )
    cleaned["arrears_longest_duration_m_utilities"] = (
        _replace_mixed_categoricals_floats(
            float_nan_values=[9999999999, 9999999998],
            categories_nan_entries=["I don't know", "I prefer not to say"],
            series=raw[f"ci{column_time_identifier}{column_code}"],
        )
    )

    cleaned["benefit_anw_gross_amt"] = _replace_mixed_categoricals_floats(
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know"],
        series=raw[f"ci{column_time_identifier}111"],
    )

    col_code_benefit_anw_gross_total_amount_categ = (
        _handle_inconsistent_column_code_in_raw(
            112,
            368,
            2014,
            survey_time_index[source_file_name],
        )
    )
    cleaned["benefit_anw_gross_amt_categ"] = _replace_rename_categorical_column(
        raw[
            f"ci{column_time_identifier}{col_code_benefit_anw_gross_total_amount_categ}"
        ],
        renaming_dict={
            "i don't know": pd.NA,
            "i don\x92t know": pd.NA,
            "i prefer not to say": pd.NA,
            "less than 1,000 euros": "< 1,000",
            "1,000-3,000 euros": "1,000-3,000",
            "3,000-6,000 euros": "3,000-6,000",
            "6,000-12,000 euros": "6,000-12,000",
            "12,000-30,000 euros": "12,000-30,000",
            "less than 4,000 euros": "< 4,000",
            "4,000-8,000 euros": "4,000-8,000",
            "12,000-16,000 euros": "12,000-16,000",
            "8,000-12,000 euros": "8,000-12,000",
            "16,000-20,000 euros": "16,000-20,000",
        },
        is_ordered=False,
    )

    cleaned["benefit_anw_net_amt"] = _replace_mixed_categoricals_floats(
        **_handle_missing_column(raw, f"ci{column_time_identifier}113"),
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
    )

    cleaned["benefit_healthcare_net_amt"] = _replace_mixed_categoricals_floats(
        **_handle_missing_column(raw, f"ci{column_time_identifier}143"),
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
    )

    cleaned["benefit_inval_gross_amt"] = _replace_mixed_categoricals_floats(
        **_handle_missing_column(raw, f"ci{column_time_identifier}143"),
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
    )
    cleaned["benefit_inval_gross_amt_categ"] = _replace_rename_categorical_column(
        raw[f"ci{column_time_identifier}114"],
        renaming_dict={
            "i don't know": pd.NA,
            "i don\x92t know": pd.NA,
            "i prefer not to say": pd.NA,
            "less than 1,000 euros": "< 1,000",
            "1,000-3,000 euros": "1,000-3,000",
            "3,000-6,000 euros": "3,000-6,000",
            "6,000-12,000 euros": "6,000-12,000",
            "12,000-30,000 euros": "12,000-30,000",
        },
        is_ordered=False,
    )

    series = _handle_missing_column(raw, f"ci{column_time_identifier}137")["series"]
    cleaned["benefit_inval_net_amt"] = _apply_lowest_float_dtype(
        _replace_missing_floats(series, [9999999999, 9999999998]),
    )

    series = _handle_missing_column(raw, f"ci{column_time_identifier}126")["series"]
    cleaned["benefit_ioaw_gross_amt"] = _apply_lowest_float_dtype(
        _replace_missing_floats(series, [9999999999, 9999999998]),
    )

    cleaned["benefit_ioaw_gross_amt_categ"] = _replace_rename_categorical_column(
        **_handle_missing_column(raw, f"ci{column_time_identifier}127"),
        renaming_dict={
            "i don't know": pd.NA,
            "i don\x92t know": pd.NA,
            "i prefer not to say": pd.NA,
            "less than 1,000 euros": "< 1,000",
            "1,000-3,000 euros": "1,000-3,000",
            "3,000-6,000 euros": "3,000-6,000",
            "6,000-12,000 euros": "6,000-12,000",
            "12,000-30,000 euros": "12,000-30,000",
        },
        is_ordered=False,
    )

    cleaned["benefit_ioaw_net_amt"] = _replace_mixed_categoricals_floats(
        series=raw[f"ci{column_time_identifier}128"],
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
    )

    series = _handle_missing_column(raw, f"ci{column_time_identifier}334")["series"]
    cleaned["benefit_iow_gross_amt"] = _apply_lowest_float_dtype(
        _replace_missing_floats(series, [9999999999, 9999999998]),
    )

    col_name = _handle_inconsistent_column_code_in_raw(
        335,
        371,
        2014,
        survey_time_index[source_file_name],
    )
    cleaned["benefit_iow_gross_amt_categ"] = _replace_rename_categorical_column(
        **_handle_missing_column(
            raw,
            f"ci{column_time_identifier}{col_name}",
        ),
        renaming_dict={
            "i don't know": pd.NA,
            "i don\x92t know": pd.NA,
            "i prefer not to say": pd.NA,
            "less than 1,000 euros": "< 1,000",
            "1,000-3,000 euros": "1,000-3,000",
            "3,000-6,000 euros": "3,000-6,000",
            "6,000-12,000 euros": "6,000-12,000",
            "12,000-30,000 euros": "12,000-30,000",
        },
        is_ordered=True,
    )

    cleaned["benefit_iow_net_amt"] = _replace_mixed_categoricals_floats(
        **_handle_missing_column(raw, f"ci{column_time_identifier}336"),
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
    )

    cleaned["benefit_kindgebonden_net_amt"] = _replace_mixed_categoricals_floats(
        **_handle_missing_column(raw, f"ci{column_time_identifier}330"),
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
    )

    cleaned["benefit_orp_pens_gross_amt"] = _replace_mixed_categoricals_floats(
        series=raw[f"ci{column_time_identifier}117"],
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=["I don't know", "I prefer not to say"],
    )

    col_name = _handle_inconsistent_column_code_in_raw(
        256,
        379,
        2019,
        survey_time_index[source_file_name],
    )
    cleaned["chance_to_lose_job"] = _replace_mixed_categoricals_floats(
        float_nan_values=[9999999999, 9999999998],
        categories_nan_entries=[
            998,
            999,
            "NaN",
            "n/a since I am voluntarily quitting my job",
            "n/a since I don\x92t have a job",
        ],
        series=raw[f"ci{column_time_identifier}{col_name}"],
    )

    return cleaned
