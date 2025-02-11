"""Tasks to perform extra cleaning steps on datasets produced from raw files."""

import importlib
from typing import Annotated

import pandas as pd
from pytask import DataCatalog, task

from liss_cleaning.config import SRC_EXTRA_DATASETS_CLEANING
from liss_cleaning.raw_datasets_cleaning.task_clean_datasets import (
    CATALOG_STACKED_DATASETS,
)


def get_cleaning_function_from_dataset_module(dataset_name):
    """Import the cleaning function from the dataset module."""
    import_string = (
        str(SRC_EXTRA_DATASETS_CLEANING).split("src/")[1].replace("/", ".")
        + ".cleaners."
        + dataset_name
    )
    module = importlib.import_module(import_string)
    return module.clean_dataset


CATALOGS_EXTRA_DATASETS = {
    "matching_probabilities": ["ambiguous_beliefs_stacked"],
    "yearly_background_variables": ["monthly_background_variables_stacked"],
    # "some_panel": ["dataset_1", "dataset_2"], # noqa: ERA001
}

FINAL_DATASETS = DataCatalog(name="final_datasets")

for final_dataset_name, source_datasets in CATALOGS_EXTRA_DATASETS.items():
    func = get_cleaning_function_from_dataset_module(final_dataset_name)

    @task(id=f"make_{final_dataset_name}")
    def task_make_new_dataset(
        function=func,
        source_datasets=[CATALOG_STACKED_DATASETS[n] for n in source_datasets],
    ) -> Annotated[pd.DataFrame, FINAL_DATASETS[final_dataset_name]]:
        """Make a new dataset from the cleaned datasets."""
        return function(*source_datasets)
