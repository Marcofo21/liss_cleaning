"""All the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
SRC_DATA = SRC / "data"
BLD = SRC.joinpath("../..", "bld").resolve()
BLD_COLS_METADATA = BLD / "data_structures" / "variables_metadata"
BLD_CLEANED_DATA = BLD / "cleaned_data"
DATA_STRUCTURE_SOURCES = SRC.joinpath("structures_sources").resolve()

NORMALIZED_FORMAT = "parquet"
PANEL_FORMAT = "csv"

TEST_DIR = SRC.joinpath("..", "tests").resolve()

DATASETS_TO_CLEAN = [
    "001-background-variables",
    "002-health",
    "010-economic-situation-income",
    "009-economic-situation-assets",
    "monthly_background_variables",
    "xxx-ambiguous-beliefs",
]

DATASETS_TO_PRODUCE = [
    "economic_situation_income",
    "economic_situation_assets",
    "monthly_background_variables",
    "yearly_background_variables",
    "ambiguous_beliefs",
]

PANELS_TO_MAKE = {
    "left_tail_exploratory": {
        "economic_situation_income": "ALL",
        "economic_situation_assets": "ALL",
        "yearly_background_variables": "ALL",
        "panel_time_index": "year",
        "save_null": True,
    },
}


__all__ = [
    "BLD",
    "BLD_COLS_METADATA",
    "DATASETS_TO_CLEAN",
    "DATASETS_TO_PRODUCE",
    "DATA_STRUCTURE_SOURCES",
    "NORMALIZED_FORMAT",
    "PANELS_TO_MAKE",
    "PANEL_FORMAT",
    "SRC",
    "TEST_DIR",
]
