"""Task to produce panel dataset including variables from different sources."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task

from liss_cleaning.config import (
    BLD,
    NORMALIZED_FORMAT,
    PANEL_FORMAT,
    PANELS_TO_MAKE,
)
from liss_cleaning.data_cleaning.make_panel.produce_panel import merge_data
from liss_cleaning.helper_modules.general_error_handlers import (
    _check_file_exists,
    _check_variables_exist,
)
from liss_cleaning.helper_modules.load_save import load_data, save_data

for panel, datasets_variables in PANELS_TO_MAKE.items():

    @task
    def task_make_panel(
        variables_to_include: dict = datasets_variables,
        datasets_to_merge: dict = {
            dataset: BLD / "cleaned_data" / f"{dataset}.{NORMALIZED_FORMAT}"
            for dataset in datasets_variables.keys() - ["panel_time_index", "save_null"]
        },
        merge_config: dict = PANEL_FORMAT,
        produces: Annotated[Path, Product] = BLD
        / "panel_dataset"
        / f"{panel}.{PANEL_FORMAT}",
    ):
        """Make panel dataset including variables from different sources."""
        data_to_merge = []
        time_index = variables_to_include.pop("panel_time_index")
        save_null = variables_to_include.pop("save_null")
        for dataset, variables in variables_to_include.items():
            _check_file_exists(datasets_to_merge[dataset])
            data = load_filtered_data(dataset, datasets_to_merge, variables)
            data_to_merge.append(data)

        panel_data = merge_data(data_to_merge, time_index, save_null)
        save_data(panel_data, produces)


def load_filtered_data(
    dataset: str, depends_on: dict, variables: list | str
) -> pd.DataFrame:
    """Load data for the merging process."""
    data = load_data(depends_on[dataset])
    if variables == "ALL":
        return data
    if not isinstance(variables, list):
        raise TypeError(
            "Items in PANELS_TO_MAKE dictionaries must be a list or an 'ALL' string."
        )

    _check_variables_exist(data, variables)
    return data[variables]
