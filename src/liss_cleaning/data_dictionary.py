import os

from liss_cleaning.config import BLD_CLEANED_DATA, SRC


def _get_source_product_files(dataset_name):
    """Get the source files paths from the dictionary in the cleaning module."""
    module = __import__(
        f"liss_cleaning.data_cleaning.make_normalized_datasets.specific_cleaners.{dataset_name}_cleaner",
        fromlist=[f"{dataset_name}"],
    )
    module_dict = module.dependencies_time_index
    dependencies_names = list(module_dict.keys())
    product_paths = []
    for name in dependencies_names:
        if name == "index_name":
            continue
        product_paths.append(_make_product_path(module_dict[name], dataset_name))
    return dict(zip(dependencies_names, product_paths, strict=False))


def _make_product_path(time_identifier, dataset_name):
    """Make the path for the product file."""
    return BLD_CLEANED_DATA / dataset_name / f"{dataset_name}_{time_identifier}.arrow"


def _get_dataset_names():
    """Get the names of the datasets."""
    file_names = os.listdir(
        SRC / "data_cleaning" / "make_normalized_datasets" / "specific_cleaners"
    )
    file_names = [name for name in file_names if name.endswith(".py")]
    file_names = [name for name in file_names if name != "__init__.py"]
    file_names = [name.replace(".py", "") for name in file_names]
    file_names = [name.replace("_cleaner", "") for name in file_names]
    file_names = [name for name in file_names if name != "template"]
    return file_names


data_dictionary = {
    dataset_name: _get_source_product_files(dataset_name)
    for dataset_name in _get_dataset_names()
}
