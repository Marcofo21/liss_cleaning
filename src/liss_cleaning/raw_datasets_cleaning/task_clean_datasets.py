"""Task to make individual cleaned datasets."""

import importlib
from typing import Annotated

import pandas as pd
from pytask import DataCatalog, task

from liss_cleaning.config import SRC_DATA, SRC_RAW_DATASETS_CLEANING
from liss_cleaning.helper_modules.load_save import load_data


def get_cleaning_function_from_dataset_module(dataset_name):
    """Import the cleaning function from the dataset module."""
    import_string = (
        str(SRC_RAW_DATASETS_CLEANING).split("src/")[1].replace("/", ".")
        + ".cleaners."
        + dataset_name
        + "_cleaner"
    )
    module = importlib.import_module(import_string)
    return module.clean_dataset


RAW_PATHS = {
    "ambiguous_beliefs": [
        SRC_DATA / "xxx-ambiguous-beliefs/wave-1/L_gaudecker2018_1_6p.dta",
        SRC_DATA / "xxx-ambiguous-beliefs/wave-2/L_gaudecker2018_2_6p.dta",
        SRC_DATA / "xxx-ambiguous-beliefs/wave-3/L_gaudecker2019_3_6p.dta",
        SRC_DATA / "xxx-ambiguous-beliefs/wave-4/L_gaudecker2019_4_6p.dta",
        SRC_DATA / "xxx-ambiguous-beliefs/wave-5/L_gaudecker2020_5_6p.dta",
        SRC_DATA / "xxx-ambiguous-beliefs/wave-6/L_gaudecker2020_6_6p.dta",
        SRC_DATA / "xxx-ambiguous-beliefs/wave-7/L_gaudecker2021_7_6p.dta",
    ],
}

CATALOG_CLEANED_INDIVIDUAL_DATASETS = DataCatalog(name="individual_cleaned_datasets")

CATALOG_STACKED_DATASETS = DataCatalog(name="stacked_datasets")

for cleaner_module_name, paths_to_raw_files in RAW_PATHS.items():
    func = get_cleaning_function_from_dataset_module(cleaner_module_name)
    for path_to_raw_data in paths_to_raw_files:

        @task(id=f"clean_{path_to_raw_data.stem}")
        def task_clean_one_dataset(
            path=path_to_raw_data,
            function=func,
            script_path=SRC_RAW_DATASETS_CLEANING
            / "cleaners"
            / f"{cleaner_module_name}_cleaner.py",
        ) -> Annotated[
            pd.DataFrame,
            CATALOG_CLEANED_INDIVIDUAL_DATASETS[f"{path_to_raw_data.stem}_cleaned"],
        ]:
            raw = load_data(path)
            return function(raw, str(path).split("/")[-1])

    @task(id=f"stack_{cleaner_module_name}")
    def task_stack_datasets(
        cleaned_datasets=[
            CATALOG_CLEANED_INDIVIDUAL_DATASETS[f"{path_to_raw_data.stem}_cleaned"]
            for path_to_raw_data in paths_to_raw_files
        ],
        dataset_name=cleaner_module_name,
    ) -> Annotated[
        pd.DataFrame, CATALOG_STACKED_DATASETS[f"{cleaner_module_name}_stacked"]
    ]:
        return pd.concat(cleaned_datasets)
