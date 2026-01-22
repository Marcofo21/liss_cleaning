"""Tasks to perform extra cleaning steps on datasets produced from raw files."""

from typing import Annotated

import pandas as pd
from pytask import DataCatalog, task

from liss_cleaning.config import SRC_EXTRA_DATASETS_CLEANING
from liss_cleaning.make_final_datasets.cleaners import (
    matching_probabilities,
    yearly_background_variables,
)
from liss_cleaning.raw_datasets_cleaning.task_clean_datasets import (
    CATALOG_STACKED_DATASETS,
)


CLEANER_MODULES = {
    "matching_probabilities": matching_probabilities,
    "yearly_background_variables": yearly_background_variables,
}

CATALOGS_EXTRA_DATASETS = {
    "matching_probabilities": ["ambiguous_beliefs_stacked"],
    "yearly_background_variables": [
        "monthly_background_variables_stacked",
        "economic_situation_assets_stacked",
    ],
}

FINAL_DATASETS = DataCatalog(name="final_datasets")

for final_dataset_name, source_datasets in CATALOGS_EXTRA_DATASETS.items():
    cleaner_module = CLEANER_MODULES[final_dataset_name]

    @task(id=f"make_{final_dataset_name}")
    def task_make_new_dataset(
        function=cleaner_module.clean_dataset,
        source_datasets=[CATALOG_STACKED_DATASETS[n] for n in source_datasets],
        script=SRC_EXTRA_DATASETS_CLEANING / "cleaners" / f"{final_dataset_name}.py",
    ) -> Annotated[pd.DataFrame, FINAL_DATASETS[final_dataset_name]]:
        """Make a new dataset from the cleaned datasets."""
        return function(*source_datasets)
