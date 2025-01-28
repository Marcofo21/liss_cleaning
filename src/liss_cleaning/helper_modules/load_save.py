import pandas as pd
import yaml


def save_data(df, path):
    """Function to save a dataset depending on the format."""
    if str(path).endswith(".pickle"):
        df.to_pickle(path)
    elif str(path).endswith(".csv"):
        df.to_csv(path, index=False)
    elif str(path).endswith(".dta"):
        df.to_stata(path, write_index=False)
    elif str(path).endswith(".parquet"):
        df.to_parquet(path)
    else:
        msg = f"Format {path.suffix} not supported."
        raise ValueError(msg)


def load_data(path):
    """Function to load a dataset depending on the format."""
    extension = str(path).split(".")[-1]
    if extension == "pickle":
        return pd.read_pickle(path)
    if extension == "csv":
        return pd.read_csv(path)
    if extension == "dta":
        return pd.read_stata(path)
    if extension == "parquet":
        return pd.read_parquet(path)
    msg = f"Format {extension} not supported."
    raise ValueError(msg)


def read_yaml(path):
    """Read a YAML file.

    Args:
        path (str or pathlib.Path): Path to file.

    Returns:
        dict: The parsed YAML file.

    """
    with path.open() as stream:
        try:
            out = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            info = f"""The YAML file could not be loaded. Please check that {path}
            points to a valid YAML file."""
            raise ValueError(info) from error
    return out
