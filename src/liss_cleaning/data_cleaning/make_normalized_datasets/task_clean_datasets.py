"""Task to make individual cleaned datasets."""

import importlib
from pathlib import Path

import pandas as pd
from pytask import task

from liss_cleaning.config import BLD, NORMALIZED_FORMAT, SRC
from liss_cleaning.data_dictionary import data_dictionary
from liss_cleaning.helper_modules.load_save import load_data, save_data


def _get_cleaning_function_from_dataset_module(dataset_name):
    """Get the cleaning function from the specific cleaner module."""
    module = importlib.import_module(
        f"liss_cleaning.data_cleaning.make_normalized_datasets.specific_cleaners.{dataset_name}_cleaner"
    )
    return module.clean_dataset


to_produce = data_dictionary.keys()

for new_dataset in to_produce:
    for source_file in data_dictionary[new_dataset]:
        path_to_produce = data_dictionary[new_dataset][source_file]
        cleaning_func = _get_cleaning_function_from_dataset_module(new_dataset)

        @task(id=f"clean_{new_dataset}_{source_file.stem}")
        def task_clean_dataset(
            source_path: Path = source_file,
            produces=path_to_produce,
            cleaning_func=cleaning_func,
            cleaning_script=SRC
            / "data_cleaning"
            / "make_normalized_datasets"
            / "specific_cleaners"
            / f"{new_dataset}_cleaner.py",
        ):
            """Clean the dataset."""
            raw = load_data(source_path)
            cleaned = cleaning_func(raw, source_path)
            save_data(cleaned, produces)


for new_dataset in to_produce:
    dep_dictionary = data_dictionary[new_dataset]

    def task_merge_survey(
        data_paths=dep_dictionary,
        produces=BLD / "merged_waves" / f"{new_dataset}.{NORMALIZED_FORMAT}",
    ):
        """Merge the cleaned datasets."""
        df = pd.DataFrame()
        for dataset in data_paths.values():
            df = pd.concat([df, load_data(dataset)], axis=0)
        save_data(df, produces)
