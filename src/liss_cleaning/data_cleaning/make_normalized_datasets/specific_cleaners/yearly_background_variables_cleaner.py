import pandas as pd

from liss_cleaning.config import BLD
from liss_cleaning.helper_modules.general_cleaners import (
    _apply_lowest_float_dtype,
    _apply_lowest_int_dtype,
)

dependencies_time_index = {
    BLD / "merged_waves" / "monthly_background_variables.arrow": "all_years",
    "index_name": "year",
}


def clean_dataset(
    raw,
    source_file_name,  # noqa: ARG001
) -> pd.DataFrame:
    raw = raw.reset_index(drop=False)
    df = pd.DataFrame()

    df["personal_id"] = _apply_lowest_int_dtype(raw["personal_id"])
    df["year"] = pd.to_numeric(raw["year_month"].apply(lambda x: x.split("-")[0]))
    df = df.drop_duplicates(subset=["personal_id", "year"], keep="first")
    raw["year"] = pd.to_numeric(raw["year_month"].apply(lambda x: x.split("-")[0]))

    df["age"] = _apply_lowest_int_dtype(
        _get_median_for_index(raw, ["personal_id", "year"], "age").round()
    )

    ## to_do: test properly and fix mode function to replace some get first
    df["age_cbs"] = _get_first_for_index(raw, ["personal_id", "year"], "age_cbs")
    df["age_cbs"] = df["age_cbs"].astype("category")

    df["birth_year"] = _apply_lowest_int_dtype(
        _get_first_for_index(raw, ["personal_id", "year"], "birth_year")
    )

    df["dom_situation"] = _get_first_for_index(
        raw, ["personal_id", "year"], "dom_situation"
    )
    df["dom_situation"] = df["dom_situation"].astype("category")

    df["dwelling_type"] = _get_first_for_index(
        raw, ["personal_id", "year"], "dwelling_type"
    )
    df["dwelling_type"] = df["dwelling_type"].astype("category")

    df["education_cbs"] = _get_first_for_index(
        raw, ["personal_id", "year"], "education_cbs"
    )
    df["education_cbs"] = df["education_cbs"].astype("category")

    df["education_irrespective_diploma"] = _get_first_for_index(
        raw, ["personal_id", "year"], "education_irrespective_diploma"
    )
    df["education_irrespective_diploma"] = df["education_irrespective_diploma"].astype(
        "category"
    )

    df["gender"] = _get_first_for_index(raw, ["personal_id", "year"], "gender")
    df["gender"] = df["gender"].astype("category")

    df["gross_income_cat"] = _get_first_for_index(
        raw, ["personal_id", "year"], "gross_income_cat"
    )
    df["gross_income_cat"] = df["gross_income_cat"].astype("category")

    df["gross_income_hh"] = _apply_lowest_float_dtype(
        _get_mean_for_index(raw, ["personal_id", "year"], "gross_income_hh")
    )

    df["gross_income_imputed_personal"] = _apply_lowest_float_dtype(
        _get_mean_for_index(
            raw, ["personal_id", "year"], "gross_income_imputed_personal"
        )
    )

    df["gross_income_incl_cat"] = _get_first_for_index(
        raw, ["personal_id", "year"], "gross_income_incl_cat"
    )
    df["gross_income_incl_cat"] = df["gross_income_incl_cat"].astype("category")

    df["hh_children"] = _get_first_for_index(
        raw, ["personal_id", "year"], "hh_children"
    )
    df["hh_children"] = df["hh_children"].astype("category")

    df["hh_head_age"] = _apply_lowest_int_dtype(
        _get_median_for_index(raw, ["personal_id", "year"], "hh_head_age").round()
    )

    df["hh_id"] = _apply_lowest_int_dtype(
        _get_first_for_index(raw, ["personal_id", "year"], "hh_id")
    )

    df["hh_members"] = _get_first_for_index(raw, ["personal_id", "year"], "hh_members")
    df["hh_members"] = df["hh_members"].astype("category")

    df["respondent_position_hh"] = _get_first_for_index(
        raw, ["personal_id", "year"], "respondent_position_hh"
    )
    df["respondent_position_hh"] = df["respondent_position_hh"].astype("category")

    df["hh_sim_computer"] = _get_first_for_index(
        raw, ["personal_id", "year"], "hh_sim_computer"
    )
    df["hh_sim_computer"] = df["hh_sim_computer"].astype("category")

    df["hh_head_lives_partner"] = _get_first_for_index(
        raw, ["personal_id", "year"], "hh_head_lives_partner"
    )
    df["hh_head_lives_partner"] = df["hh_head_lives_partner"].astype("category")

    df["net_income_cat"] = _get_first_for_index(
        raw, ["personal_id", "year"], "net_income_cat"
    )
    df["net_income_cat"] = df["net_income_cat"].astype("category")

    df["net_income_hh"] = _apply_lowest_float_dtype(
        _get_mean_for_index(raw, ["personal_id", "year"], "net_income_hh")
    )

    df["net_income_imputed_personal"] = _apply_lowest_float_dtype(
        _get_mean_for_index(raw, ["personal_id", "year"], "net_income_imputed_personal")
    )

    for col in [
        "net_income_incl_cat",
        "net_income_personal",
        "occupation",
        "origin",
    ]:
        df[col] = _get_first_for_index(raw, ["personal_id", "year"], col)
        df[col] = df[col].astype("category")

    return df


def _get_median_for_index(df, index, column):
    """Get the median value for a column grouped by an index.

    Args:
            df (pd.DataFrame): The dataframe to group.
            index (str|list): The column(s) to group by.
            column (str): The column to get the median for.

    Returns:
            pd.Series: The median values for the column grouped by the index.
    """
    return df.dropna(subset=[column]).groupby(index)[column].transform("median")


def _get_first_for_index(df, index, column):
    """Get the first value for a column grouped by an index.

    Args:
            df (pd.DataFrame): The dataframe to group.
            index (str|list): The column(s) to group by.
            column (str): The column to get the first value for.

    Returns:
            pd.Series: The first values for the column grouped by the index.
    """
    return df.dropna(subset=[column]).groupby(index)[column].transform("first")


def _get_most_common_for_index(df, index, column):
    """Get the most common value for a column grouped by an index.

    Args:
            df (pd.DataFrame): The dataframe to group.
            index (str|list): The column(s) to group by.
            column (str): The column to get the most common value for.

    Returns:
            pd.Series: The most common value for the column grouped by the index.
    """
    return (
        df.dropna(subset=[column]).groupby(index)[column].transform(lambda x: x.mode())
    )


def _get_mean_for_index(df, index, column):
    """Get the mean value for a column grouped by an index.

    Args:
            df (pd.DataFrame): The dataframe to group.
            index (str|list): The column(s) to group by.
            column (str): The column to get the mean for.

    Returns:
            pd.Series: The mean values for the column grouped by the index.
    """
    return df.dropna(subset=[column]).groupby(index)[column].transform("mean")
