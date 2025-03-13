"""Task to clean indidual raw files for each survey, and stack them in a dataset."""

import importlib
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import DataCatalog, task

from liss_cleaning.config import SRC_DATA, SRC_RAW_DATASETS_CLEANING
from liss_cleaning.helper_modules.load_save import load_data


def get_cleaning_function_from_dataset_module(dataset_name):
    """Import the cleaning function from the corresponding dataset module."""
    import_string = (
        str(SRC_RAW_DATASETS_CLEANING).split("liss-cleaning/src/")[1].replace("/", ".")
        + ".cleaners."
        + dataset_name
        + "_cleaner"
    )
    module = importlib.import_module(import_string)
    return module.clean_dataset


def get_dta_files_from_folder(folder_path: Path) -> list[Path]:
    """Get all the .dta files from a folder and its subfolders,
    ignoring files whose stem ends with '_do_not_use'.
    """
    return [p for p in folder_path.rglob("*.dta") if not p.stem.endswith("_do_not_use")]


RAW_PATHS = {
    "ambiguous_beliefs": get_dta_files_from_folder(SRC_DATA / "xxx-ambiguous-beliefs"),
    "monthly_background_variables": get_dta_files_from_folder(
        SRC_DATA / "001-background-variables"
    ),
    "health": get_dta_files_from_folder(SRC_DATA / "002-health"),
    "economic_situation_assets": get_dta_files_from_folder(
        SRC_DATA / "009-economic-situation-assets"
    ),
    "economic_situation_income": get_dta_files_from_folder(
        SRC_DATA / "010-economic-situation-income"
    ),
}


CATALOG_CLEANED_INDIVIDUAL_DATASETS = DataCatalog(name="individual_cleaned_datasets")

CATALOG_STACKED_DATASETS = DataCatalog(name="stacked_datasets")

for survey_name, paths_to_raw_files in RAW_PATHS.items():
    func = get_cleaning_function_from_dataset_module(survey_name)
    for path_to_raw_data in paths_to_raw_files:

        @task(id=f"clean_{survey_name}_{path_to_raw_data.stem}")
        def task_clean_one_dataset(
            path=path_to_raw_data,
            function=func,
            script_path=SRC_RAW_DATASETS_CLEANING
            / "cleaners"
            / f"{survey_name}_cleaner.py",
        ) -> Annotated[
            pd.DataFrame,
            CATALOG_CLEANED_INDIVIDUAL_DATASETS[f"{path_to_raw_data.stem}_cleaned"],
        ]:
            """Clean raw data from one wave of a survey."""
            raw = load_data(path)
            return function(raw, str(path).split("/")[-1])

    @task(id=f"stack_{survey_name}")
    def task_stack_datasets(
        cleaned_datasets=[
            CATALOG_CLEANED_INDIVIDUAL_DATASETS[f"{path_to_raw_data.stem}_cleaned"]
            for path_to_raw_data in paths_to_raw_files
        ],
        dataset_name=survey_name,
    ) -> Annotated[pd.DataFrame, CATALOG_STACKED_DATASETS[f"{survey_name}_stacked"]]:
        """Stack all the cleaned waves for each survey."""
        return pd.concat(handle_empty_columns(cleaned_datasets))


def handle_empty_columns(cleaned_datasets):
    """Remove empty columns from the dataset before concatenation."""
    return [df.dropna(axis=1, how="all") for df in cleaned_datasets]
