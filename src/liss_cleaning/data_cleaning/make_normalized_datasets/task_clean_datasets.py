"""Task to make individual cleaned datasets."""

from pathlib import Path
from typing import Annotated

from liss_data_cleaning.config import BLD, DATASETS_TO_PRODUCE, NORMALIZED_FORMAT, SRC
from liss_data_cleaning.data_cleaning.make_normalized_datasets.cleaner import (
    clean_dataset,
)
from liss_data_cleaning.helper_modules.load_save import save_data
from pytask import Product, task


def get_source_files_dict(new_dataset):
    """Get the source files paths from the dictionary in the cleaning module."""
    module = __import__(
        f"liss_data_cleaning.data_cleaning.make_normalized_datasets.specific_cleaners.{new_dataset}_cleaner",
        fromlist=[f"{new_dataset}"],
    )
    return module.survey_time_index


for new_dataset in DATASETS_TO_PRODUCE:

    @task
    def task_clean_dataset(
        dataset_name=new_dataset,
        source_files_index_info=get_source_files_dict(new_dataset),
        source_files=list(get_source_files_dict(new_dataset).keys()),
        general_cleaner_script=SRC
        / "data_cleaning"
        / "make_normalized_datasets"
        / "cleaner.py",
        specific_cleaner_script=SRC
        / "data_cleaning"
        / "make_normalized_datasets"
        / "specific_cleaners"
        / f"{new_dataset}_cleaner.py",
        cleaner_helpers_script=SRC / "helper_modules" / "general_cleaners.py",
        cleaned_dataset: Annotated[Path, Product] = BLD
        / "cleaned_data"
        / f"{new_dataset}.{NORMALIZED_FORMAT}",
    ):
        """Clean the dataset."""
        cleaned = clean_dataset(dataset_name, source_files_index_info)
        save_data(cleaned, cleaned_dataset)
