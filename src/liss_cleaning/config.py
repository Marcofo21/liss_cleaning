"""All the general configuration of the project."""

from pathlib import Path

SRC = Path(__file__).parent.resolve()
SRC_DATA = SRC / "data"
SRC_RAW_DATASETS_CLEANING = SRC / "raw_datasets_cleaning"
BLD = SRC.joinpath("../..", "bld").resolve()
BLD_CLEANED_DATA = BLD / "individual_wave"

TEST_DIR = SRC.joinpath("..", "tests").resolve()


__all__ = [
    "BLD",
    "BLD_CLEANED_DATA",
    "SRC",
    "TEST_DIR",
]
