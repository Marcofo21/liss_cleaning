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

    # Banking assets - special handling for "has" column
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

    # Insurance assets (single-premium, life annuity, endowment)
    cleaned["has_insurance_assets"] = raw[f"ca{column_time_identifier}005"]
    cleaned["value_insurance_assets"] = _process_asset_value(
        raw, column_time_identifier, "014", "015"
    )
    cleaned["value_insurance_assets"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_insurance_assets"], cleaned["has_insurance_assets"]
    )

    # Risky assets (investments, stocks, bonds, etc.)
    cleaned["has_risky_assets"] = _replace_rename_categorical_column(
        raw[f"ca{column_time_identifier}006"], {"no": "No", "yes": "Yes"}
    )
    cleaned["value_risky_assets"] = _process_asset_value(
        raw, column_time_identifier, "016", "017"
    )
    cleaned["value_risky_assets"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_risky_assets"], cleaned["has_risky_assets"]
    )

    # Real estate (not personal home)
    cleaned["has_real_estate"] = raw[f"ca{column_time_identifier}007"]
    cleaned["value_real_estate"] = _process_asset_value(
        raw, column_time_identifier, "018", "019"
    )
    cleaned["value_real_estate"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_real_estate"], cleaned["has_real_estate"]
    )

    # Real estate mortgage debt (LIABILITY - subtract from wealth)
    cleaned["has_real_estate_mortgage"] = raw[f"ca{column_time_identifier}020"]
    cleaned["value_real_estate_mortgage"] = _process_asset_value(
        raw, column_time_identifier, "021", "022"
    )
    cleaned["value_real_estate_mortgage"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_real_estate_mortgage"], cleaned["has_real_estate_mortgage"]
    )

    # Vehicles (cars, motorcycles, boats, caravans)
    cleaned["has_vehicles"] = raw[f"ca{column_time_identifier}008"]
    cleaned["value_vehicles"] = _process_asset_value(
        raw, column_time_identifier, "023", "024"
    )
    cleaned["value_vehicles"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_vehicles"], cleaned["has_vehicles"]
    )

    # Loans to family/friends/acquaintances
    cleaned["has_loans_to_others"] = raw[f"ca{column_time_identifier}010"]
    cleaned["value_loans_to_others"] = _process_asset_value(
        raw, column_time_identifier, "025", "026"
    )
    cleaned["value_loans_to_others"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_loans_to_others"], cleaned["has_loans_to_others"]
    )

    # Other assets (antiques, jewelry, collections, cash)
    cleaned["has_other_assets"] = raw[f"ca{column_time_identifier}011"]
    cleaned["value_other_assets"] = _process_asset_value(
        raw, column_time_identifier, "027", "028"
    )
    cleaned["value_other_assets"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_other_assets"], cleaned["has_other_assets"]
    )

    # Private company equity (for DGA - majority-shareholder directors)
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

    # Partnership equity
    cleaned["has_partnership"] = raw[f"ca{column_time_identifier}080"]
    cleaned["partnership_fiscal_year_matches_calendar"] = raw.get(
        f"ca{column_time_identifier}041", np.nan
    )
    cleaned["value_partnership_equity"] = _process_asset_value(
        raw, column_time_identifier, "083", "084"
    )
    cleaned["value_partnership_equity"] = _add_zeros_for_nas_in_indicator_column(
        cleaned["value_partnership_equity"], cleaned["has_partnership"]
    )

    # Calculate total wealth
    cleaned["total_wealth"] = _calculate_total_wealth(cleaned)

    return cleaned


def _calculate_total_wealth(cleaned: pd.DataFrame) -> pd.Series:
    """Calculate total wealth by summing assets and subtracting liabilities."""
    asset_columns = [
        "value_banking_assets",
        "value_insurance_assets",
        "value_risky_assets",
        "value_real_estate",
        "value_vehicles",
        "value_loans_to_others",
        "value_other_assets",
        "value_private_company_equity",
        "value_partnership_equity",
    ]

    liability_columns = [
        "value_real_estate_mortgage",
    ]

    total_assets = cleaned[asset_columns].sum(axis=1)
    total_liabilities = cleaned[liability_columns].sum(axis=1)
    return total_assets - total_liabilities


def _process_asset_value(raw, column_time_identifier, value_code, categorical_code):
    """Process asset value column with missing value replacement and categorical imputation."""
    value_col = _replace_missing_floats(
        raw[f"ca{column_time_identifier}{value_code}"],
        float_nan_values=[
            np.nan,
            9999999999.0,
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
    mask = (indicator_column.isin(["No", 2, "2"])) & (result.isna())
    result.loc[mask] = 0.0
    return result


def _add_imputed_values_from_categorical_column(
    main_column: pd.Series, impute_from_column: pd.Series
) -> pd.Series:
    """Impute missing values in main_asset_column from impute_from_column."""
    try:
        impute_from_column_str = impute_from_column.astype(str)
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
                -9.0: np.nan,
                -9: np.nan,
                "999": np.nan,
                "999.0": np.nan,
                999: np.nan,
                "I don't know": np.nan,
            }
        )
        assets_from_categoricals = assets_from_categoricals.astype(float)
        if (assets_from_categoricals < 0).any():
            raise ValueError(
                "Negative values found in imputed assets from categoricals."
            )
        return main_column.fillna(assets_from_categoricals)
    except Exception as e:
        print(f"Error in imputing values from categorical column: {e}")
        return main_column
