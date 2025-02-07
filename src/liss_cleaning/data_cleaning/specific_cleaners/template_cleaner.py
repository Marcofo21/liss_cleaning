import pandas as pd

# create a clean_name_of_the_dataset function that cleans a single wave of the data
# (one file at the time)
# you can use the metadata as we do or just specify whichever cleaning within the
# module (just ignore the arguments
# for the metadata)

dependencies_time_index = {
    "name_dataset.dta": 2016,
    "name_dataset_2.dta": 2017,
    "index_name": "year",
}


def clean_dataset(raw, dta_file) -> pd.DataFrame:
    pass
