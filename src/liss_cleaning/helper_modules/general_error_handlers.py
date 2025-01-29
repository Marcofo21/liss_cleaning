from pathlib import Path

import pandas as pd


def _check_variables_exist(data: pd.DataFrame, variables: list):
    """Check if variables exist in the data, if not, give error message."""
    for variable in variables:
        if variable not in data.columns:
            msg = f"Variable {variable} not in data columns."
            raise ValueError(msg)


def _check_object_type(obj, obj_type):
    """Checks if the object is of the expected type.

    Args:
        obj: the object to check.
        obj_type: the expected type of the object.

    Returns:
        None
    Raises:
        TypeError: if the object is not of the expected type.
    """
    if not isinstance(obj, obj_type):
        msg = f"Expected {obj} to be of type {obj_type}, got {type(obj)}"
        raise TypeError(msg)


def _check_file_exists(file_path):
    """Check if file exists, if it does not, give error message."""
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        msg = f"File {file_path} does not exist."
        # if the file_path points to
        raise FileNotFoundError(msg)


def _check_object_type(obj, obj_type):
    """Checks if the object is of the expected type.

    Args:
        obj: the object to check.
        obj_type: the expected type of the object.

    Returns:
        None
    Raises:
        TypeError: if the object is not of the expected type.
    """
    if not isinstance(obj, obj_type):
        msg = f"Expected {obj} to be of type {obj_type}, got {type(obj)}"
        raise TypeError(msg)


def _check_series_dtype(series, dtype):
    """Checks if the series has the expected dtype.

    Args:
        series(pd.Series): the series to check.
        dtype: the expected dtype of the series.

    Returns:
        None
    Raises:
        TypeError: if the series does not have the expected dtype.
    """
    if series.dtype != dtype:
        msg = f"Expected series to have dtype {dtype}, got {series.dtype}"
        raise TypeError(msg)
