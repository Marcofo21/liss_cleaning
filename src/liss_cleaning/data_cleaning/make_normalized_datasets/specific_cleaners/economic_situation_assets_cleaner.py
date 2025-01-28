import numpy as np
import pandas as pd
from liss_data_cleaning.config import SRC_DATA
from liss_data_cleaning.helper_modules.general_cleaners import (
    _apply_lowest_int_dtype,
    _handle_inconsistent_column_code_in_raw,
    _replace_missing_floats,
    _replace_rename_categorical_column,
)

pd.set_option("future.no_silent_downcasting", True)

survey_time_index = {
    f"{SRC_DATA}/009-economic-situation-assets/wave-1/ca08a_1.0p_EN.dta": 2008,
    f"{SRC_DATA}/009-economic-situation-assets/wave-2/ca10b_EN_1.0p.dta": 2010,
    f"{SRC_DATA}/009-economic-situation-assets/wave-3/ca12c_EN_1.0p.dta": 2012,
    f"{SRC_DATA}/009-economic-situation-assets/wave-4/ca14d_2.0p_EN.dta": 2014,
    f"{SRC_DATA}/009-economic-situation-assets/wave-5/ca16e_EN_1.0p.dta": 2016,
    f"{SRC_DATA}/009-economic-situation-assets/wave-6/ca18f_EN_1.0p.dta": 2018,
    f"{SRC_DATA}/009-economic-situation-assets/wave-7/ca20g_EN_1.0p.dta": 2020,
    "index_name": "year",
}


def clean_economic_situation_assets(raw, source_file_name) -> pd.DataFrame:
    cleaned = pd.DataFrame(index=raw.index)
    column_time_identifier = source_file_name.split("/")[-1].split("_")[0][2:5]
    cleaned["personal_id"] = _apply_lowest_int_dtype(raw["nomem_encr"])
    cleaned[survey_time_index["index_name"]] = survey_time_index[source_file_name]

    col_name = _handle_inconsistent_column_code_in_raw(
        str(4).zfill(3), str(1).zfill(3), 2010, survey_time_index[source_file_name]
    )
    cleaned["has_banking_assets"] = raw[f"ca{column_time_identifier}{col_name}"]
    cleaned["has_risky_assets"] = _replace_rename_categorical_column(
        raw["ca" + column_time_identifier + "006"], {"no": "No", "yes": "Yes"}
    )
    cleaned["value_risky_assets"] = _replace_missing_floats(
        raw["ca" + column_time_identifier + "016"],
        float_nan_values=[np.nan, 9999999999.0, 99999999998.0],
    )

    return cleaned
