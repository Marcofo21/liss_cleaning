from liss_cleaning.config import BLD_CLEANED_DATA


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


data_dictionary = {
    "economic_situation_assets": _get_source_product_files("economic_situation_assets"),
}
