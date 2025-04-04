import pandas as pd

from liss_cleaning.helper_modules.general_cleaners import (
    _apply_lowest_int_dtype,
)


def clean_dataset(
    raw: pd.DataFrame,
    source_file_name: str,
) -> pd.DataFrame:
    """Clean the corona questionnaire data.

    Args:
        raw (pd.DataFrame): The raw data to clean.
        source_file_name (str): The name of the source file.

    Returns:
        pd.DataFrame: The cleaned data.
    """
    cleaned_data = pd.DataFrame(index=raw.index)
    cleaned_data["personal_id"] = _apply_lowest_int_dtype(raw["nomem_encr"])
    cleaned_data["date"] = pd.to_datetime(raw["DatumB"], format="mixed", dayfirst=True)
    if "wave" not in source_file_name:
        wave_id = 2 if "4.0" in source_file_name else 1
    elif "Macro" in source_file_name:
        wave_id = 7
    else:
        wave_id = int(str(source_file_name).split("wave")[1].split("_")[0])
    cleaned_data["wave"] = wave_id

    if wave_id == 2:
        cleaned_data["pr_AEX_gt_1100"] = raw["forw_look_1_nocheck"]
        cleaned_data["pr_AEX_gt_950_lt_1100"] = raw["forw_look_2_nocheck"]
        cleaned_data["pr_AEX_lt_950"] = raw["forw_look_3_nocheck"]

        cleaned_data["pr_AEX_gt_1100_consistent"] = raw["forw_look_1"]
        cleaned_data["pr_AEX_gt_950_lt_1100_consistent"] = raw["forw_look_2"]
        cleaned_data["pr_AEX_lt_950_consistent"] = raw["forw_look_3"]
        cleaned_data["treatment_assignment"] = pd.Categorical(
            raw["arandom"].map(
                {
                    1: "Control",
                    2: "Upward trend treatment",
                    3: "Downward trend treatment",
                }
            ),
            categories=[
                "Upward trend treatment",
                "Downward trend treatment",
                "Control",
            ],
            ordered=False,
        )

    return cleaned_data
