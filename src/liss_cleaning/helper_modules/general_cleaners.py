import warnings

import numpy as np
import pandas as pd


def _handle_missing_column(df: pd.DataFrame, column_name: str) -> dict:
    """Checks whether column exists in the dataset, and if it does not, creates
    a new column with the same name and fills it with NaNs.

    Args:
        df(pd.DataFrame): the data frame to check.
        column_name(str): the name of the column to check.

    Returns:
        dict: a dictionary with the series and a boolean indicating whether the column
        is missing. Contains: series(pd.Series), is_missing(bool).
    """
    if column_name not in df.columns:
        return {
            "series": pd.Series([pd.NA] * len(df), name=column_name),
            "is_missing": True,
        }
    return {
        "series": df[column_name],
        "is_missing": False,
    }


def _replace_values(value, replacing_dict):
    """Replaces values in a series.

    Args:
        value(any): the value to replace.
        replacing_dict(dict): the dictionary with the values to replace.

    Returns:
        series(pd.Series): the series with the replaced values.
    """
    return replacing_dict.get(value, value)


def _replace_mixed_categoricals_floats(
    float_nan_values: list,
    categories_nan_entries: list,
    series: pd.Series,
    is_missing: bool = False,
) -> pd.Series:
    """Clean columns that have either categorical or float values in different datasets,
    and convert them to float.

    Args:
        series(pd.Series): the series to clean.
        is_missing(bool): whether the column is missing in the dataset. This is used to
        handle loops over survey waves that may not have a query. See notes.
        float_nan_values(list): the values to convert to NaN.
        categories_nan_entries(list): the categories to convert to NaN.

    Returns:
        series(pd.Series): the cleaned series.

    Notes:
        Some surveys are missing specific columns in one wave or another. You can use
        _handle_missing_column to handle this case in the modules.

    """
    if not is_missing:
        if series.dtype == "float64":
            series = _replace_missing_floats(series, float_nan_values)
            return _apply_lowest_float_dtype(series)
        if series.dtype == "category":
            if not set(categories_nan_entries).issubset(set(series.cat.categories)):
                categories_nan_entries = set(categories_nan_entries).intersection(
                    set(series.cat.categories),
                )
            return _categorical_to_float(series, categories_nan_entries)
        return None
    if is_missing:
        return series
    return None


def _categorical_to_float(series: pd.Series, nan_entries: list) -> pd.Series:
    """Convert a categorical series to float.

    Args:
        series(pd.Series): the series to convert.
        nan_entries(list): the categories to convert to NaN.
        changing_categories(bool): whether some of the categories miss in some waves.

    Returns:
        series(pd.Series): the converted series.
    """
    nan_entries = _handle_changing_na_categories(series, nan_entries)
    series = series.cat.remove_categories(nan_entries)
    series = series.apply(lambda x: pd.to_numeric(x))
    return _apply_lowest_float_dtype(series)


def _handle_changing_na_categories(series: pd.Series, nan_entries: list) -> pd.Series:
    if not set(nan_entries).issubset(set(series.unique())):
        series_values = series.unique()
        series_values = [x for x in series_values if x is not pd.NA]
        nan_entries = [x for x in nan_entries if x in series_values]
    return nan_entries


def _replace_rename_categorical_column(
    series: pd.Series,
    renaming_dict: dict,
    is_ordered: bool = False,
    is_missing: bool = False,
) -> pd.Series:
    """Replace and rename the values of a categorical column. Put all the files in
       lower case before replacing, so that dictionary is case insensitive.
       Also handle cases where the column is missing in the dataset.

    Args:
        series(pd.Series): the series to replace and rename.
        renaming_dict(dict): the dictionary with the values to replace and rename.
        is_ordered(bool): whether the categories should be ordered.
        is_missing(bool): whether the column is missing in the dataset. This is used to
        handle loops over survey waves that may not have a query. Use
        _handle_missing_column if this is the case-. Otherwise, if the query is present
        in all survey waves, set to False.

    Returns:
        series(pd.Series): the series with the replaced and renamed values.
    """
    if not is_missing:
        series = series.astype(str)
        series = series.str.lower()
        series = series.map(renaming_dict)
        new_categories = set(renaming_dict.values()) - {pd.NA} - {np.nan}
        old_categories_not_renamed = (
            set(series.unique()) - new_categories - {pd.NA} - {np.nan}
        )
        if len(old_categories_not_renamed) > 0:
            warnings.warn(
                f"Categories {old_categories_not_renamed} from the raw data "
                "not found in the renaming dictionary. "
                "The missing categories will become pd.NA, check if this is intended.",
                stacklevel=2,
            )
        series = pd.Categorical(series, categories=new_categories, ordered=is_ordered)
        return pd.Series(series)
    return series


def _handle_inconsistent_column_code_in_raw(
    pre_change_code: int | str,
    post_change_code: int | str,
    year_switch: int,
    year_current_df: int,
) -> int:
    """Handle the case where the column code in the raw data changes between survey
    waves.
    """
    if int(year_current_df) >= int(year_switch):
        return str(post_change_code)
    return str(pre_change_code)


def _replace_missing_floats(series: pd.Series, float_nan_values: list) -> pd.Series:
    """Replace missing floats in a series.

    Args:
        series(pd.Series): the series to replace missing floats in.
        float_nan_values(list): the values to replace with NaN.

    Returns:
        series(pd.Series): the series with the missing floats replaced.
    """
    for value in float_nan_values:
        series = series.replace(value, pd.NA)
    return series


def _find_lowest_int_dtype(sr: pd.Series) -> str:
    """Find the lowest integer dtype for a series.

    Args:
        sr (pd.Series): The series to check.

    Returns:
        str: The lowest integer dtype.

    """
    if "float" in sr.dtype.name:
        sr = sr.astype("float[pyarrow]")
    if sr.min() >= 0:
        if sr.max() <= 255:
            return "uint8[pyarrow]"
        if sr.max() <= 65535:
            return "uint16[pyarrow]"
        if sr.max() <= 4294967295:
            return "uint32[pyarrow]"
        return "uint64[pyarrow]"
    if sr.min() >= -128 and sr.max() <= 127:
        return "int8[pyarrow]"
    if sr.min() >= -32768 and sr.max() <= 32767:
        return "int16[pyarrow]"
    if sr.min() >= -2147483648 and sr.max() <= 2147483647:
        return "int32[pyarrow]"
    return "int64[pyarrow]"


def _apply_lowest_int_dtype(sr: pd.Series) -> str:
    """Apply the lowest integer dtype to a series."""
    return sr.astype(_find_lowest_int_dtype(sr))


def _find_lowest_float_dtype(sr: pd.Series) -> str:
    """Find the lowest float dtype for a series.

    Args:
        sr (pd.Series): The series to check.

    Returns:
        str: The lowest float dtype.

    """
    if sr.isna().any():
        return "float64[pyarrow]"
    if sr.min() >= 0:
        if sr.max() <= 3.4028235e38:
            return "float32[pyarrow]"
        return "float64[pyarrow]"
    if sr.min() >= -1.7976931348623157e308 and sr.max() <= 1.7976931348623157e308:
        return "float64[pyarrow]"
    return "float64[pyarrow]"


def _check_if_np_nan_or_pd_na(sr: pd.Series):
    """Check if the series contains np.nan or pd.NA."""
    array = np.array(sr)
    if np.isnan(array).any():
        raise ValueError("The series contains np.nan.")


def _apply_lowest_float_dtype(sr: pd.Series) -> str:
    """Apply the lowest integer dtype to a series."""
    return sr.astype(_find_lowest_float_dtype(sr))
