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
    banking_has_col_name = _handle_inconsistent_column_code_in_raw(
        str(4).zfill(3),
        str(1).zfill(3),
        2010,
        cleaned["year"].unique(),
    )
    cleaned["has_banking_assets"] = raw[
        f"ca{column_time_identifier}{banking_has_col_name}"
    ]
    cleaned["value_banking_assets"] = _process_asset_value(
        raw, column_time_identifier, "012", "013"
    )

    cleaned["has_insurance_assets"] = raw[f"ca{column_time_identifier}005"]
    cleaned["value_insurance_assets"] = _process_asset_value(
        raw, column_time_identifier, "014", "015"
    )
    cleaned["value_insurance_assets"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_insurance_assets"], cleaned["has_insurance_assets"]
    )

    cleaned["has_risky_assets"] = _replace_rename_categorical_column(
        raw[f"ca{column_time_identifier}006"], {"no": "No", "yes": "Yes"}
    )
    cleaned["value_risky_assets"] = _process_asset_value(
        raw, column_time_identifier, "016", "017"
    )
    cleaned["value_risky_assets"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_risky_assets"], cleaned["has_risky_assets"]
    )

    cleaned["has_real_estate"] = raw[f"ca{column_time_identifier}007"]
    cleaned["value_real_estate"] = _process_asset_value(
        raw, column_time_identifier, "018", "019"
    )
    cleaned["value_real_estate"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_real_estate"], cleaned["has_real_estate"]
    )

    cleaned["has_real_estate_mortgage"] = raw[f"ca{column_time_identifier}020"]
    cleaned["value_real_estate_mortgage"] = _process_asset_value(
        raw, column_time_identifier, "021", "022"
    )
    cleaned["value_real_estate_mortgage"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_real_estate_mortgage"], cleaned["has_real_estate_mortgage"]
    )

    cleaned["has_vehicles"] = raw[f"ca{column_time_identifier}008"]
    cleaned["value_vehicles"] = _process_asset_value(
        raw, column_time_identifier, "023", "024"
    )
    cleaned["value_vehicles"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_vehicles"], cleaned["has_vehicles"]
    )

    cleaned["has_loans_to_others"] = raw[f"ca{column_time_identifier}010"]
    cleaned["value_loans_to_others"] = _process_asset_value(
        raw, column_time_identifier, "025", "026"
    )
    cleaned["value_loans_to_others"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_loans_to_others"], cleaned["has_loans_to_others"]
    )

    cleaned["has_other_assets"] = raw[f"ca{column_time_identifier}011"]
    cleaned["value_other_assets"] = _process_asset_value(
        raw, column_time_identifier, "027", "028"
    )
    cleaned["value_other_assets"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_other_assets"], cleaned["has_other_assets"]
    )

    if column_time_identifier in ["08a"]:
        cleaned["is_dga"] = pd.Series(np.nan, index=raw.index)
    else:
        cleaned["is_dga"] = raw[f"ca{column_time_identifier}079"]
    cleaned["has_private_pension_company"] = raw.get(
        f"ca{column_time_identifier}030", np.nan
    )
    cleaned["private_company_stake_percentage"] = raw.get(
        f"ca{column_time_identifier}034", np.nan
    )
    cleaned["value_private_company_equity"] = _process_asset_value(
        raw, column_time_identifier, "035", "036"
    )
    cleaned["value_private_company_equity"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_private_company_equity"], cleaned["is_dga"]
    )

    cleaned["has_partnership"] = raw.get(f"ca{column_time_identifier}080", np.nan)
    cleaned["partnership_fiscal_year_matches_calendar"] = raw.get(
        f"ca{column_time_identifier}041", np.nan
    )
    if column_time_identifier in ["10b", "08a"]:
        cleaned["value_partnership_equity"] = pd.Series(np.nan, index=raw.index)
    else:
        cleaned["value_partnership_equity"] = _process_asset_value(
            raw, column_time_identifier, "083", "084"
        )
    cleaned["value_partnership_equity"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_partnership_equity"], cleaned["has_partnership"]
    )

    cleaned["total_wealth"] = _calculate_total_wealth(cleaned, source_file_name)
    cleaned["share_risky_assets"] = np.where(
        cleaned["total_wealth"] == 0,
        0,
        cleaned["value_risky_assets"] / cleaned["total_wealth"].replace(0, np.nan),
    )
    return cleaned


def _check_column_sanity(
    cleaned: pd.DataFrame,
    column_name: str,
    source_file_name: str,
    allow_negative: bool = False,
) -> None:
    """
    Check that a column contains only valid numeric values.

    Args:
        cleaned: DataFrame containing the column
        column_name: Name of the column to check
        source_file_name: Name of the source file (for error messages)
        allow_negative: Whether negative values are allowed

    Raises:
        TypeError: If column contains non-numeric values
        ValueError: If column contains negative values when not allowed
    """
    if column_name not in cleaned.columns:
        raise KeyError(
            f"Column '{column_name}' not found in dataset '{source_file_name}'"
        )

    col = cleaned[column_name]

    if col.dtype == "object":
        string_mask = col.apply(lambda x: isinstance(x, str))
        if string_mask.any():
            string_values = col.loc[string_mask].unique()
            raise TypeError(
                f"Column '{column_name}' in dataset '{source_file_name}' contains string values. "
                f"Sample values: {list(string_values[:5])}"
            )

    test_numeric = pd.to_numeric(col, errors="coerce")
    non_null_before = col.notna().sum()
    non_null_after = test_numeric.notna().sum()

    if non_null_after < non_null_before:
        problem_values = col.loc[col.notna() & test_numeric.isna()].unique()
        raise TypeError(
            f"Column '{column_name}' in dataset '{source_file_name}' contains "
            f"non-numeric values. Sample problematic values: {list(problem_values[:5])}"
        )

    if not allow_negative:
        if (test_numeric < 0).any():
            negative_count = (test_numeric < 0).sum()
            negative_sample = test_numeric[test_numeric < 0].head(5).tolist()
            raise ValueError(
                f"Column '{column_name}' in dataset '{source_file_name}' contains "
                f"{negative_count} negative values (not allowed). "
                f"Sample negative values: {negative_sample}"
            )


def _calculate_total_wealth(cleaned: pd.DataFrame, source_file_name: str) -> pd.Series:
    """Calculate total wealth by summing assets and subtracting liabilities."""

    assets_allowing_negative = {
        "value_banking_assets": True,
        "value_insurance_assets": False,
        "value_risky_assets": True,
        "value_real_estate": False,
        "value_vehicles": False,
        "value_loans_to_others": False,
        "value_other_assets": False,
        "value_private_company_equity": True,
        "value_partnership_equity": True,
    }

    liability_columns = [
        "value_real_estate_mortgage",
    ]

    for col, allow_negative in assets_allowing_negative.items():
        _check_column_sanity(
            cleaned, col, source_file_name, allow_negative=allow_negative
        )

    for col in liability_columns:
        _check_column_sanity(cleaned, col, source_file_name, allow_negative=False)

    asset_columns = list(assets_allowing_negative.keys())
    total_assets = cleaned[asset_columns].sum(axis=1)
    total_liabilities = cleaned[liability_columns].sum(axis=1)

    return total_assets - total_liabilities


def _process_asset_value(raw, column_time_identifier, value_code, categorical_code):
    """Process asset value column with missing value replacement and categorical imputation."""
    value_col = raw[f"ca{column_time_identifier}{value_code}"].copy()

    if value_col.dtype.name == "category":
        value_col = value_col.astype(str)

    if value_col.dtype == "object":
        value_col = value_col.replace(
            {
                "I don't know": np.nan,
                "I prefer not to say": np.nan,
                "nan": np.nan,
            }
        )
        value_col = pd.to_numeric(value_col, errors="coerce")

    value_col = _replace_missing_floats(
        value_col,
        float_nan_values=[
            np.nan,
            9999999999.0,
            99999999998.0,
            99999999999.0,
            9999999998.0,
            -9999999999.0,
            -9999999998.0,
            -8,
            -9,
        ],
    )

    return _add_imputed_values_from_categorical_column(
        value_col,
        raw[f"ca{column_time_identifier}{categorical_code}"],
    )


def _add_zeros_for_nas_in_indicator_column(
    main_asset_column: pd.Series, indicator_column: pd.Series
) -> pd.Series:
    """Impute zeros in main_asset_column where indicator_column is 'No' or 2."""
    result = main_asset_column.copy()
    indicator_str = indicator_column.astype(str)
    mask = (indicator_str.isin(["No", "2", "2.0"])) & (result.isna())
    result[mask] = 0.0
    return result


def _add_imputed_values_from_categorical_column(
    main_column: pd.Series, impute_from_column: pd.Series
) -> pd.Series:
    """Impute missing values in main_asset_column from impute_from_column."""
    try:
        impute_from_column_str = impute_from_column.astype(str)
        impute_from_column_str = impute_from_column_str.str.replace(
            "\x80", "€", regex=False
        )

        assets_from_categoricals = impute_from_column_str.replace(
            {
                "less than  50": 25.0,
                "50 to  250": 150.0,
                "250 to  500": 375.0,
                "500 to  750": 625.0,
                "750 to  1,000": 875.0,
                "1,000 to  2,500": 1750.0,
                "2,500 to  5,000": 3750.0,
                "5,000 to  7,500": 6250.0,
                "7,500 to  10,000": 8750.0,
                "10,000 to  11,500": 10750.0,
                "11,500 to  14,000": 12750.0,
                "14,000 to  17,000": 15500.0,
                "17,000 to  20,000": 18500.0,
                "20,000 to  25,000": 22500.0,
                "25,000 or more": 25000.0,
                "less than € 50": 25.0,
                "€ 50 to € 250": 150.0,
                "€ 250 to € 500": 375.0,
                "€ 500 to € 750": 625.0,
                "€ 750 to € 1,000": 875.0,
                "€ 1,000 to € 2,500": 1750.0,
                "€ 2,500 to € 5,000": 3750.0,
                "€ 5,000 to € 7,500": 6250.0,
                "€ 7,500 to € 10,000": 8750.0,
                "€ 10,000 to € 11,500": 10750.0,
                "€ 11,500 to € 14,000": 12750.0,
                "€ 14,000 to € 17,000": 15500.0,
                "€ 17,000 to € 20,000": 18500.0,
                "€ 20,000 to € 25,000": 22500.0,
                "€ 25,000 or more": 25000.0,
                "less than € 500": 250.0,
                "€ 500 to € 1,500": 1000.0,
                "€ 1,500 to € 2,500": 2000.0,
                "€ 2,500 to € 5,000": 3750.0,
                "€ 10,000 to € 12,000": 11000.0,
                "€ 12,000 to € 15,000": 13500.0,
                "€ 15,000 to € 20,000": 17500.0,
                "€ 20,000 to € 25,000": 22500.0,
                "€ 25,000 to € 50,000": 37500.0,
                "€ 25,550 to € 50,000": 37775.0,
                "€ 25,500 to € 50,000": 37750.0,
                "€ 50,000 to € 75,000": 62500.0,
                "€ 75,000 to € 100,000": 87500.0,
                "€ 100,000 or more": 100000.0,
                "less than € 50,000": 25000.0,
                "€ 50,000 to € 100,000": 75000.0,
                "€  50,000 to € 100,000": 75000.0,
                "€ 100,000 to € 150,000": 125000.0,
                "€ 150,000 to € 200,000": 175000.0,
                "€ 200,000 to € 250,000": 225000.0,
                "€ 250,000 to € 400,000": 325000.0,
                "€ 400,000 to € 500,000": 450000.0,
                "€ 500,000 to € 1,000,000": 750000.0,
                "€ 1,000,000 to € 2,500,000": 1750000.0,
                "€ 2,500,000 or more": 2500000.0,
                "positive, but smaller than € 50,000": 25000.0,
                "negative": np.nan,
                "-9.0": np.nan,
                "-9": np.nan,
                "-150000.0": np.nan,
                "-200.0": np.nan,
                "-10000.0": np.nan,
                "-70000.0": np.nan,
                "-1180.0": np.nan,
                -9.0: np.nan,
                -9: np.nan,
                -150000.0: np.nan,
                -200.0: np.nan,
                -10000.0: np.nan,
                -70000.0: np.nan,
                -1180.0: np.nan,
                "999": np.nan,
                "999.0": np.nan,
                999: np.nan,
                999.0: np.nan,
                "I don't know": np.nan,
                "I prefer not to say": np.nan,
                "nan": np.nan,
                "-5000000.0": np.nan,
                "-500000.0": np.nan,
                "-45000.0": np.nan,
                "-88000.0": np.nan,
            }
        )
        assets_from_categoricals = pd.to_numeric(
            assets_from_categoricals, errors="coerce"
        )

        if (assets_from_categoricals < 0).any():
            raise ValueError(
                "Negative values found in imputed assets from categoricals."
            )
        return main_column.fillna(assets_from_categoricals)
    except Exception as e:
        print(f"Error in imputing values from categorical column: {e}")
        return main_column
