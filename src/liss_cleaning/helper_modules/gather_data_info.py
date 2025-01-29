import pandas as pd

from liss_cleaning.helper_modules.general_error_handlers import _check_object_type


def extract_info_each_column(data):
    """Extracts information about each column in the dataset.

    Args:
        data(pd.DataFrame): the dataset to extract information from.

    Returns:
        dict: the dictionary with information about each column.
    """
    info = {}
    for column in data.columns:
        info[column] = {
            "dtype": str(data[column].dtype),
            "number_missing_values": str(data[column].isna().sum()),
            "number_unique_values": str(data[column].nunique()),
            "unique_values": str(data[column].unique()),
            "number_non-nan_values": str(data[column].count()),
            # if contains "prefer not to say" or "I don\x92t know
            "number prefer not to say": str(
                data[column]
                .astype(str)
                .str.lower()
                .str.contains("i prefer not to say")
                .sum(),
            ),
            "number don't know": str(
                data[column].astype(str).str.lower().str.contains("i don't know").sum(),
            ),
            "number don\x92t know": str(
                data[column]
                .astype(str)
                .str.lower()
                .str.contains("i don\x92t know")
                .sum(),
            ),
            "number 9999999999": str(
                data[column].astype(str).str.contains("9999999999").sum(),
            ),
            "number 9999999998": str(
                data[column].astype(str).str.contains("9999999998").sum(),
            ),
        }
    return info


def make_new_mapping(old_mapping):
    """Creates a new mapping dictionary for the variables in the dataset.

    Args:
        old_mapping(pd.Dataframe): the old mapping dictionary (keys: old variable names,
        values: new variable names).

    Returns:
        new_mapping(dict): the new mapping dictionary (keys: old variable names, values:
        new variable names).
    """
    _check_old_mapping(old_mapping)

    new_mapping = {}

    columns_referring_to_datasets = _get_columns_referring_to_datasets(
        old_mapping.columns,
    )
    old_mapping[columns_referring_to_datasets] = old_mapping[
        columns_referring_to_datasets
    ].fillna("Variable not in here/not mapped yet")

    for _index, row in old_mapping.iterrows():
        if pd.isna(row["new_name"]):
            continue
        new_name = row["new_name"]
        new_mapping[new_name] = {}
        for column in columns_referring_to_datasets:
            new_mapping[new_name][column] = row[column]
            if "labels" in old_mapping.columns:
                new_mapping[new_name]["description"] = row["labels"]

    _check_new_mapping(new_mapping)

    return new_mapping


def _check_old_mapping(old_mapping):
    """Checks if the old mapping is valid.

    Args:
        old_mapping(pd.DataFrame): the old mapping dictionary (keys: old variable names,
        values: new variable names).
    """
    _check_object_type(old_mapping, pd.DataFrame)

    if "new_name" not in old_mapping.columns:
        msg = "Expected column 'new_name' in old_mapping."
        raise ValueError(msg)


def _check_new_mapping(new_mapping):
    """Checks if the new mapping is valid.

    Args:
        new_mapping(dict): the new mapping dictionary (keys: old variable names, values:
        new variable names).
    """
    _check_object_type(new_mapping, dict)
    if not new_mapping:
        msg = "Expected new_mapping to be non empty."
        raise ValueError(msg)


def _get_columns_referring_to_datasets(columns):
    """Returns the columns that refer to the dataset.

    Args:
        columns(list): the columns of the dataset.

    Returns:
        columns_referring_to_dataset(list): the columns that refer to the dataset.
    """
    return [column for column in columns if column.endswith(".dta")]
