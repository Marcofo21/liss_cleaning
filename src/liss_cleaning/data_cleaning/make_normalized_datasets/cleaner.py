import importlib
import warnings

import pandas as pd

from liss_cleaning.helper_modules.general_error_handlers import (
    _check_file_exists,
)
from liss_cleaning.helper_modules.load_save import load_data


def clean_dataset(dataset_name, source_files_index_info) -> pd.DataFrame:
    """Clean the dataset by performing in each column the cleaning operations
    specified in the columns_cleaning_metadata.

    Args:
        dataset_name(str): The name of the new dataset (used to point at the specific
            cleaner module).
        source_files_index_info(dict): The dictionary with the information about the
            source files' paths and the time index.

    Returns:
        pandas.DataFrame: the cleaned data frame.
    """
    cleaning_function = _get_cleaning_function_from_dataset_module(dataset_name)
    datasets_to_clean_dict = {
        key: value
        for key, value in source_files_index_info.items()
        if key != "index_name"
    }
    for raw_file in datasets_to_clean_dict:
        _check_file_exists(raw_file)
    cleaned_datasets = {}
    try:
        for raw_file in datasets_to_clean_dict:
            raw = load_data(raw_file)
            cleaned = cleaning_function(raw, raw_file)
            cleaned = cleaned.set_index(
                ["personal_id", source_files_index_info["index_name"]],
            )
            cleaned_datasets[raw_file] = cleaned
        del cleaning_function
        cleaned_data = _squash_datasets(cleaned_datasets.values())
        _check_index_is_unique(cleaned_data)
    except (KeyError, ValueError, TypeError) as e:
        error_msg = f"An error occurred while cleaning the dataset {dataset_name}: {e}"
        raise CleaningError(error_msg) from e

    return cleaned_data


class CleaningError(Exception):
    pass


def _squash_datasets(cleaned_datasets):
    """Squash the cleaned datasets into one, so that there is one obs for each id and
    nas are replaced by the first non-na value in the group.
    """
    # ignore warning in this function
    if len(cleaned_datasets) == 1:
        return cleaned_datasets[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            pd.concat(cleaned_datasets, ignore_index=False)
            .groupby(level=[0, 1])
            .first()
        )


def _get_time_index_dictionary_from_dataset_module(dataset_name):
    """Get the time index dictionary from the module."""
    module = __import__(
        (
            "liss_cleaning.data_cleaning.make_normalized_datasets"
            ".specific_cleaners.{dataset_name}_cleaner"
        ),
        fromlist=[f"{dataset_name}"],
    )
    return module.survey_time_index


def _check_raw_datasets_non_empty(raw_datasets):
    """Check if the raw datasets dictionary is non-empty."""
    if not raw_datasets:
        msg = "The raw datasets dictionary is empty."
        raise ValueError(msg)


def _is_it_wave_or_date(waves_mapping) -> bool:
    """Check if the mapping is for waves or dates."""
    if all(isinstance(wave, int) for wave in waves_mapping):
        return True
    if all(isinstance(wave, str) for wave in waves_mapping):
        return False
    msg = "The mapping does not contain the wave or date column."
    raise ValueError(msg)


def _check_old_cols_present(merged_raws, cols_mapping) -> None:
    """Check if the old columns names are present in the raw data."""
    for key in cols_mapping:
        for raw_dta, raw_df in merged_raws.items():
            col_to_check = cols_mapping[key][raw_dta]
            if col_to_check == "Variable not in here/not mapped yet":
                continue
            if col_to_check not in raw_df.columns:
                msg = f"The column {col_to_check} is not present in the data."
                raise ValueError(
                    msg,
                )


def _get_cleaning_function_from_dataset_module(dataset_name):
    """Get the cleaning function from the module.

    Args:
        dataset_name (str): The name of the dataset for which to retrieve the cleaning
            function.

    Returns:
        function: The cleaning function associated with the dataset.

    Raises:
        ImportError: If the module for the dataset cannot be imported.
        AttributeError: If the cleaning function is not found in the module.
    """
    try:
        # Dynamically import the module based on the dataset name
        module_name = (
            "liss_cleaning.data_cleaning.make_normalized_datasets"
            f".specific_cleaners.{dataset_name}_cleaner"
        )
        dataset_module = importlib.import_module(module_name)

        function_name = f"clean_{dataset_name}"
        cleaning_function = getattr(dataset_module, function_name)

    except ImportError as e:
        error_msg = f"Module for dataset '{dataset_name}' could not be imported: {e}"
        raise ImportError(error_msg) from e

    except AttributeError as e:
        error_msg = (
            f"Cleaning function not found in module for dataset '{dataset_name}': {e}"
        )
        raise AttributeError(error_msg) from e

    return cleaning_function


def _check_all_raws_same_columns(cleaned_datasets):
    """Check if all the raws have the same columns."""
    first = next(iter(cleaned_datasets.values()))
    for cleaned in cleaned_datasets.values():
        if not first.columns.equals(cleaned.columns):
            msg = (
                "The columns of the cleaned files are not"
                " the same for the file {dta_file}."
            )
            raise ValueError(
                msg,
            )


def _check_index_is_unique(cleaned_data):
    """Check if the index is unique."""
    if not cleaned_data.index.is_unique:
        msg = "The index is not unique."
        raise ValueError(msg)
