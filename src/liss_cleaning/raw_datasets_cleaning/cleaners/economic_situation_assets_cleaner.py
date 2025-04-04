import numpy as np
import pandas as pd

from liss_cleaning.helper_modules.general_cleaners import (
    _apply_lowest_int_dtype,
    _handle_inconsistent_column_code_in_raw,
    _replace_missing_floats,
    _replace_rename_categorical_column,
)

pd.set_option("future.no_silent_downcasting", True)


def clean_dataset(raw, source_file_name) -> pd.DataFrame:
    cleaned = pd.DataFrame(index=raw.index)
    column_time_identifier = str(source_file_name).split("/")[-1].split("_")[0][2:5]
    cleaned["personal_id"] = _apply_lowest_int_dtype(raw["nomem_encr"])
    cleaned["year"] = int(f"20{column_time_identifier[0:2]}")

    col_name = _handle_inconsistent_column_code_in_raw(
        str(4).zfill(3),
        str(1).zfill(3),
        2010,
        cleaned["year"].unique(),
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
