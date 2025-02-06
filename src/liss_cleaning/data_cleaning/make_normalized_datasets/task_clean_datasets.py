"""Task to make individual cleaned datasets."""

import importlib
from typing import Annotated

import pandas as pd
from pytask import DataCatalog, PickleNode, Product, task

from liss_cleaning.config import SRC_DATA
from liss_cleaning.helper_modules.load_save import load_data


def _get_cleaning_function_from_dataset_module(dataset_name):
    """Get the cleaning function from the specific cleaner module."""
    module = importlib.import_module(
        f"liss_cleaning.data_cleaning.make_normalized_datasets.specific_cleaners.{dataset_name}_cleaner"
    )
    return module.clean_dataset


## No advantage to using catalogs here tbh; only difference between this and the
# previous
## is that intermediate gets stored in pickle; which was achievable anyways.
## I understand the point in ML projects with insanely large dataframes, but this...
data_catalogs_individual_datasets = {
    "ambiguous_beliefs": {
        "xxx-ambiguous-beliefs/wave-1/L_gaudecker2018_1_6p.dta": DataCatalog(
            name="ambiguous_beliefs_wave_1"
        ),
        "xxx-ambiguous-beliefs/wave-2/L_gaudecker2018_2_6p.dta": DataCatalog(
            name="ambiguous_beliefs_wave_2"
        ),
        "xxx-ambiguous-beliefs/wave-3/L_gaudecker2019_3_6p.dta": DataCatalog(
            name="ambiguous_beliefs_wave_3"
        ),
        "xxx-ambiguous-beliefs/wave-4/L_gaudecker2019_4_6p.dta": DataCatalog(
            name="ambiguous_beliefs_wave_4"
        ),
        "xxx-ambiguous-beliefs/wave-5/L_gaudecker2020_5_6p.dta": DataCatalog(
            name="ambiguous_beliefs_wave_5"
        ),
        "xxx-ambiguous-beliefs/wave-6/L_gaudecker2020_6_6p.dta": DataCatalog(
            name="ambiguous_beliefs_wave_6"
        ),
        "xxx-ambiguous-beliefs/wave-7/L_gaudecker2021_7_6p.dta": DataCatalog(
            name="ambiguous_beliefs_wave_7"
        ),
    },
}
# `data_catalogs_stacked_datasets` is a dictionary that contains information about
# stacked
# datasets. In this specific code snippet, it is used to store a single DataCatalog
# object
# for the stacked dataset of "ambiguous_beliefs". This DataCatalog object is named
# "ambiguous_beliefs_stacked".
data_catalogs_stacked_datasets = {
    "ambiguous_beliefs": DataCatalog(name="ambiguous_beliefs_stacked"),
}

for new_dataset, catalogs in data_catalogs_individual_datasets.items():
    for wave_data_path, catalog in catalogs.items():
        func = _get_cleaning_function_from_dataset_module(new_dataset)

        @task(id=f"clean_{new_dataset}_dataset_{wave_data_path}")
        def task_clean_one_dataset(
            path=SRC_DATA / wave_data_path,
            node: Annotated[PickleNode, Product] = data_catalogs_individual_datasets[
                new_dataset
            ][wave_data_path],
            function=func,
        ) -> Annotated[pd.DataFrame, catalog["cleaned"]]:
            raw = load_data(path)
            cleaned = function(raw, path)
            return cleaned
            # save doesn't store the pickle file??? what is this
            # it's just unnecessary complications (product files inside .pytask????)
            # EVERYONE understands paths; no one will have time to understand a class
            # so unclearly defined

    # data catalogs don't get updated between tasks LOL killer feature
    @task(id=f"stack_{new_dataset}_datasets")
    def task_stack_datasets(
        dfs=[
            data_catalogs_individual_datasets[new_dataset][wave_data_path]["cleaned"]
            for wave_data_path in catalogs
        ],
    ) -> Annotated[
        pd.DataFrame, data_catalogs_stacked_datasets[new_dataset]["cleaned"]
    ]:
        data = pd.concat(dfs)
        return data
