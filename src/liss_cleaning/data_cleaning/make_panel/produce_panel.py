import pandas as pd

# TO_DO: think a bit about what are the useful features one could implement here:
# - merging data from different sources [x]
# - handling different time indices atuomatically [?]
#   - which uses would be common? Somebody wants to use date of the survey instead of
# the year?


def merge_data(
    data_to_merge: list, panel_time_index: str, save_null: bool
) -> pd.DataFrame:
    """Merge data from different sources.

    Args:
        data_to_merge: list of dataframes to merge.
        panel_time_index: index to use for the panel dataset.
        save_null: whether to save the null values in the panel dataset.

    Returns:
        pd.DataFrame: the merged dataset.
    """
    if panel_time_index == "year":
        for n in range(1, len(data_to_merge)):
            merged_data = data_to_merge[n - 1].merge(
                data_to_merge[n], how="outer", left_index=True, right_index=True
            )
        return merged_data
    if isinstance(panel_time_index, dict):
        data_to_merge = [
            _get_consistent_index(data, panel_time_index) for data in data_to_merge
        ]
    else:
        raise TypeError(
            """panel_time_index in config.py must be either "year" (default index) or
                a dictionary specifying how to map one
                of the existing variables into an index for all datasets."""
        )
    if save_null:
        data_to_merge = [data.dropna() for data in data_to_merge]

    for n in range(1, len(data_to_merge)):
        merged_data = data_to_merge[n - 1].merge(
            data_to_merge[n], how="outer", left_index=True, right_index=True
        )
    return merged_data


def _get_consistent_index(data: pd.DataFrame, panel_time_index: dict) -> pd.DataFrame:
    """Check all indices and adjust those that are not consistent."""
    if data.index.name != panel_time_index.keys()[0]:
        data = (
            data.reset_index()
            .set_index((data["personal_id"], data[panel_time_index]))
            .drop(columns=["personal_id", panel_time_index, "year"])
        )

    return data
