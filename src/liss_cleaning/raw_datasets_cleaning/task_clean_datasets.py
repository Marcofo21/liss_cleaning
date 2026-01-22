"""Task to clean individual raw files for each survey, and stack them in a dataset."""

from typing import Annotated

import pandas as pd
from pytask import DataCatalog, task

from liss_cleaning.config import SRC_DATA, SRC_RAW_DATASETS_CLEANING
from liss_cleaning.helper_modules.general_error_handlers import _check_file_exists
from liss_cleaning.helper_modules.load_save import load_data
from liss_cleaning.raw_datasets_cleaning.cleaners import (
    ambiguous_beliefs_cleaner,
    corona_questionnaire_cleaner,
    economic_situation_assets_cleaner,
    economic_situation_income_cleaner,
    health_cleaner,
    monthly_background_variables_cleaner,
)


CLEANER_MODULES = {
    "ambiguous_beliefs": ambiguous_beliefs_cleaner,
    "monthly_background_variables": monthly_background_variables_cleaner,
    "health": health_cleaner,
    "economic_situation_assets": economic_situation_assets_cleaner,
    "economic_situation_income": economic_situation_income_cleaner,
    "corona_questionnaire": corona_questionnaire_cleaner,
}


def get_dta_files_from_folder(folder_path):
    """Get all .dta files from a folder, ignoring those ending with '_do_not_use'."""
    _check_file_exists(folder_path)
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
    "corona_questionnaire": get_dta_files_from_folder(
        SRC_DATA / "xyx-corona-questionnaire"
    ),
}


CATALOG_CLEANED_INDIVIDUAL_DATASETS = DataCatalog(name="individual_cleaned_datasets")

CATALOG_STACKED_DATASETS = DataCatalog(name="stacked_datasets")

for survey_name, paths_to_raw_files in RAW_PATHS.items():
    cleaner_module = CLEANER_MODULES[survey_name]

    for path_to_raw_data in paths_to_raw_files:

        @task(id=f"clean_{survey_name}_{path_to_raw_data.stem}")
        def task_clean_one_dataset(
            path=path_to_raw_data,
            function=cleaner_module.clean_dataset,
            script_path=SRC_RAW_DATASETS_CLEANING
            / "cleaners"
            / f"{survey_name}_cleaner.py",
        ) -> Annotated[
            pd.DataFrame,
            CATALOG_CLEANED_INDIVIDUAL_DATASETS[f"{path_to_raw_data.stem}_cleaned"],
        ]:
            """Clean raw data from one wave of a survey."""
            raw = load_data(path)
            return function(raw, path.name)

    @task(id=f"stack_{survey_name}")
    def task_stack_datasets(
        cleaned_datasets=[
            CATALOG_CLEANED_INDIVIDUAL_DATASETS[f"{p.stem}_cleaned"]
            for p in paths_to_raw_files
        ],
        dataset_name=survey_name,
    ) -> Annotated[pd.DataFrame, CATALOG_STACKED_DATASETS[f"{survey_name}_stacked"]]:
        """Stack all the cleaned waves for each survey."""
        return pd.concat(_drop_empty_columns(cleaned_datasets))


def _drop_empty_columns(dataframes):
    """Remove columns that are entirely NA before concatenation."""
    return [df.dropna(axis=1, how="all") for df in dataframes]
