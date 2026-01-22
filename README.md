# LISS data cleaning

A reproducible pipeline for cleaning and harmonizing LISS panel survey data, built with
[pytask](https://pytask-dev.readthedocs.io/).

## Overview

This project processes raw LISS (Longitudinal Internet Studies for the Social Sciences)
survey waves into analysis-ready datasets. The pipeline:

1. Cleans individual survey waves with dataset-specific cleaners
2. Stacks waves into longitudinal datasets
3. Constructs derived datasets (e.g., matching probabilities, yearly background
   variables)

## Installation

Requires [pixi](https://pixi.sh/):

```bash
pixi install
```

## Usage

Run the full pipeline:

```bash
pixi run pytask
```

Run tests:

```bash
pixi run pytest
```

## Project Structure

```
src/liss_cleaning/
├── config.py                      # Path constants
├── data/                          # Raw LISS .dta files (not tracked)
├── helper_modules/
│   ├── general_cleaners.py        # Reusable cleaning functions
│   ├── general_error_handlers.py  # Validation helpers
│   └── load_save.py               # I/O utilities
├── raw_datasets_cleaning/
│   ├── task_clean_datasets.py     # Pytask tasks for wave-level cleaning
│   └── cleaners/                  # One module per survey
│       ├── ambiguous_beliefs_cleaner.py
│       ├── monthly_background_variables_cleaner.py
│       └── ...
└── make_final_datasets/
    ├── task_extra_cleaning.py     # Pytask tasks for derived datasets
    └── cleaners/
        ├── matching_probabilities.py
        └── yearly_background_variables.py

bld/                               # Build outputs (gitignored)
tests/                             # Pytest test suite
```

## Adding a New Survey

1. Place raw `.dta` files in `src/liss_cleaning/data/<survey-folder>/`
2. Create `src/liss_cleaning/raw_datasets_cleaning/cleaners/<survey_name>_cleaner.py`
3. Implement `clean_dataset(raw, source_file_name) -> pd.DataFrame`
4. Register the survey in `RAW_PATHS` dict in `task_clean_datasets.py`

See `template_cleaner.py` for a minimal example.

## License

MIT
