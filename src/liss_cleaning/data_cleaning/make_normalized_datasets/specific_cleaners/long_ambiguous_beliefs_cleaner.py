import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

dependencies_time_index = {}


def clean_dataset(raw, source_file_name):
    source_file_name = source_file_name.split("/")[-1]
    return raw
