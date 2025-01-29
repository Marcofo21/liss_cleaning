import os

import pandas as pd

from liss_cleaning.config import SRC_DATA
from liss_cleaning.helper_modules.general_cleaners import (
    _apply_lowest_float_dtype,
    _apply_lowest_int_dtype,
    _categorical_to_float,
    _handle_missing_column,
    _replace_rename_categorical_column,
)


def _get_date_month(source_file_name):
    time_identifier = source_file_name.split("/")[-1].split("_")[1]
    return time_identifier[:4] + "-" + time_identifier[4:]


dependencies_time_index = {}
for x in os.listdir(f"{SRC_DATA}/001-background-variables"):
    if x.endswith(".dta"):
        dependencies_time_index[SRC_DATA / "001-background-variables" / x] = (
            _get_date_month(x)
        )
dependencies_time_index["index_name"] = "month"


def clean_dataset(
    raw,
    source_file_name,
) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    time_identifier = _get_date_month(str(source_file_name))

    income_categories = {
        "no income": "No income",
        "eur 500 or less": "Less than 500 euros",
        "eur 501 to eur 1000": "501-1000 euros",
        "eur 1001 to eur 1500": "1001-1500 euros",
        "eur 1501 to eur 2000": "1501-2000 euros",
        "eur 2001 to eur 2500": "2001-2500 euros",
        "eur 3001 to eur 3500": "3001-3500 euros",
        "eur 3501 to eur 4000": "3501-4000 euros",
        "eur 4001 to eur 4500": "4001-4500 euros",
        "eur 4501 to eur 5000": "4501-5000 euros",
        "eur 5001 to eur 7500": "5001-7500 euros",
        "more than eur 7500": "More than 7500 euros",
        "i really don't know": pd.NA,
        "i prefer not to say": pd.NA,
    }
    df["personal_id"] = _apply_lowest_int_dtype(raw["nomem_encr"])
    df["age"] = _apply_lowest_int_dtype(raw["leeftijd"])
    df["age_cbs"] = _replace_rename_categorical_column(
        raw["lftdcat"],
        is_ordered=True,
        renaming_dict={
            "55 - 64 years": "55-64",
            "65 years and older": "65+",
            "15 - 24 years": "18-24",
            "25 - 34 years": "25-34",
            "35 - 44 years": "35-44",
            "45 - 54 years": "45-54",
        },
    )
    df["year_month"] = time_identifier
    df["birth_year"] = _apply_lowest_int_dtype(raw["gebjaar"])
    df["civil_status"] = _replace_rename_categorical_column(
        raw["burgstat"],
        renaming_dict={
            "Married": "Married",
            "Never been married": "Never married",
            "Divorced": "Divorced",
            "Widow or widower": "Widowed",
        },
    )
    df["hh_member_participation"] = _replace_rename_categorical_column(
        raw["doetmee"],
        renaming_dict={
            "yes": "Yes",
            "no": "No",
        },
    )
    df["dom_situation"] = _replace_rename_categorical_column(
        raw["woonvorm"],
        renaming_dict={
            "(Un)married co-habitation, with child(ren)": (
                "Co-habitation, " "with child(ren)"
            ),
            "(Un)married co-habitation, without child(ren)": (
                "Co-habitation, " "without child(ren)"
            ),
            "Single, with child(ren)": "Single, with child(ren)",
            "Single": "Single",
            "Other": "Other",
            "(un)married co-habitation, with child(ren)": (
                "Co-habitation, " "with child(ren)"
            ),
            "(un)married co-habitation, without child(ren)": (
                "Co-habitation, without " "child(ren)"
            ),
            "single, with child(ren)": "Single, with child(ren)",
            "single": "Single",
            "other": "Other",
        },
    )
    df["dwelling_type"] = _replace_rename_categorical_column(
        raw["woning"],
        renaming_dict={
            "Self-owned dwelling": "Self-owned",
            "Rental dwelling": "Rental",
            "Cost-free dwelling": "Cost-free",
        },
    )
    renaming_dict_education = {
        "wo (university)": "University",
        "hbo (higher vocational education, us: college)": "Higher vocational education",
        "mbo (intermediate vocational education, us: junior college)": (
            "Intermediate " "vocational education"
        ),
        (
            "havo/vwo (higher secondary education/preparatory university education, US:"
            " senior high school)"
        ): "Higher secondary education",
        (
            "vmbo (intermediate secondary education, us: junior " "high school)"
        ): "Intermediate secondary education",
        "primary school": "Primary school",
        "other": "Other",
        "not (yet) completed any education": "Other",
        "not yet started any education": "Other",
    }
    df["education_cbs"] = _replace_rename_categorical_column(
        raw["oplcat"],
        renaming_dict=renaming_dict_education,
    )
    df["education_highest_diploma"] = _replace_rename_categorical_column(
        raw["oplmet"],
        renaming_dict=renaming_dict_education,
    )
    df["education_irrespective_diploma"] = _replace_rename_categorical_column(
        raw["oplzon"],
        renaming_dict=renaming_dict_education,
    )
    df["gender"] = _replace_rename_categorical_column(
        raw["geslacht"],
        renaming_dict={
            "Male": "Male",
            "Female": "Female",
            "male": "Male",
            "female": "Female",
        },
    )

    df["female"] = (df["gender"] == "Female").astype(int)

    df["gross_income_cat"] = _replace_rename_categorical_column(
        raw["brutocat"],
        renaming_dict=income_categories,
        is_ordered=True,
    )

    df["gross_income_hh"] = _apply_lowest_float_dtype(
        _handle_missing_column(raw, "brutohh_f")["series"],
    )
    df["gross_income_imputed_personal"] = _apply_lowest_float_dtype(
        _handle_missing_column(raw, "brutoink_f")["series"],
    )

    df["gross_income_incl_cat"] = _categorical_to_float(
        raw["brutoink"],
        nan_entries=[
            "I don't know",
            "Unknown (missing)",
            "Prefer not to say",
            "I dont know",
        ],
    )

    df["hh_children"] = _replace_rename_categorical_column(
        raw["aantalki"],
        renaming_dict={
            "None": "No children",
            "One child": "One child",
            "Two children": "Two children",
            "Three children": "Three children",
            "Four children": "Four children",
            "Five children": "Five children",
            "Six children": "Six children",
            "Seven children": "Seven children",
            "Eight children": "Eight children",
            "Nine children or more": "More than nine children",
        },
        is_ordered=True,
    )
    df["hh_head_age"] = _apply_lowest_int_dtype(
        raw["lftdhhh"],
    )
    df["hh_id"] = raw["nohouse_encr"]
    df["hh_members"] = _replace_rename_categorical_column(
        raw["aantalhh"],
        renaming_dict={
            "One person": "One person",
            "Two persons": "Two persons",
            "Three persons": "Three persons",
            "Four persons": "Four persons",
            "Five persons": "Five persons",
            "Six persons": "Six persons",
            "Seven persons": "Seven persons",
            "Eight persons": "Eight persons",
            "Nine persons or more": "More than nine persons",
        },
    )
    df["respondent_position_hh"] = _replace_rename_categorical_column(
        raw["positie"],
        renaming_dict={
            "Household head": "Household head",
            "Wedded partner": "Wedded partner",
            "Unwedded partner": "Unwedded partner",
            "Parent (in law)": "Parent (in law)",
            "Child living at home": "Child living at home",
            "Housemate": "Housemate",
            "Family member or boarder": "Family member or boarder",
            "Unknown (missing)": pd.NA,
        },
    )
    df["hh_position"] = _replace_rename_categorical_column(
        raw["positie"],
        renaming_dict={
            "Household head": "Household head",
            "Wedded partner": "Wedded partner",
            "Unwedded partner": "Unwedded partner",
            "Parent (in law)": "Parent (in law)",
            "Child living at home": "Child living at home",
            "Housemate": "Housemate",
            "Family member or boarder": "Family member or boarder",
            "Unknown (missing)": pd.NA,
        },
    )
    df["hh_sim_computer"] = _replace_rename_categorical_column(
        _handle_missing_column(raw, "simpc")["series"],
        renaming_dict={
            "yes": "Yes",
            "no": "No",
        },
    )
    df["hh_head_lives_partner"] = _replace_rename_categorical_column(
        raw["partner"],
        renaming_dict={
            "Yes": "Yes",
            "No": "No",
        },
    )
    df["urban_level_location"] = _handle_missing_column(raw, "sted")["series"]
    df["location_urban"] = _handle_missing_column(raw, "sted")["series"]
    df["net_income_cat"] = _replace_rename_categorical_column(
        raw["nettocat"],
        renaming_dict=income_categories,
        is_ordered=True,
    )
    df["net_income_hh"] = _apply_lowest_float_dtype(
        _handle_missing_column(raw, "nettohh_f")["series"],
    )
    df["has_pos_net_income"] = df["net_income_hh"] > 0
    df["net_income_imputed_personal"] = _apply_lowest_float_dtype(
        _handle_missing_column(raw, "nettoink_f")["series"],
    )
    df["net_income_incl_cat"] = _categorical_to_float(
        raw["nettoink"],
        nan_entries=[
            "I don't know",
            "Unknown (missing)",
            "Prefer not to say",
            "I dont know",
        ],
    )
    if _handle_missing_column(raw, "netinc")["is_missing"]:
        df["net_income_personal"] = pd.NA
    else:
        df["net_income_personal"] = _categorical_to_float(
            raw["netinc"],
            nan_entries=[
                "I don't know",
                "Unknown (missing)",
                "Prefer not to say",
                "I dont know",
            ],
        )
    df["occupation"] = _replace_rename_categorical_column(
        raw["belbezig"],
        renaming_dict={
            "Paid employment": "Employed",
            "Works or assists in family business": "Works in family business",
            "Autonomous professional, freelancer, or self-employed": "Self-employed",
            "Job seeker following job loss": "Job seeker (following job loss)",
            "First-time job seeker": "Job seeker (first-time)",
            "Exempted from job seeking following job loss": (
                "Exempted from job seeking" " (following job loss)"
            ),
            "Attends school or is studying": "Student",
            "Takes care of the housekeeping": "Housekeeping",
            (
                "Is pensioner ([voluntary] early retirement, " "old age pension scheme)"
            ): "Pensioner",
            "Has (partial) work disability": "Work disability",
            "Performs unpaid work while retaining unemployment benefits": (
                "Performs " "unpaid work while retaining unemployment benefits"
            ),
            "Performs voluntary work": "Voluntary work",
            "Does something else": "Other occupation",
            "Is too young to have an occupation": "Too young to have an occupation",
            "is too young to have an occupation": "Too young to have an occupation",
            "paid employment": "Employed",
            "works or assists in family business": "Works in family business",
            "autonomous professional, freelancer, or self-employed": "Self-employed",
            "job seeker following job loss": "Job seeker (following job loss)",
            "first-time job seeker": "Job seeker (first-time)",
            "exempted from job seeking following job loss": (
                "Exempted from job seeking" " (following job loss)"
            ),
            "attends school or is studying": "Student",
            "takes care of the housekeeping": "Housekeeping",
            (
                "is pensioner ([voluntary] early retirement, " "old age pension scheme)"
            ): "Pensioner",
            "has (partial) work disability": "Work disability",
            "performs unpaid work while retaining unemployment benefits": (
                "Performs " "unpaid work while retaining unemployment benefits"
            ),
            "performs voluntary work": "Voluntary work",
            "does something else": "Other occupation",
        },
    )
    df["origin"] = _replace_rename_categorical_column(
        _handle_missing_column(raw, "herkomstgroep")["series"],
        renaming_dict={
            "Dutch background": "Dutch",
            "First generation foreign, Western background": (
                "First generation " "foreign, Western"
            ),
            "First generation foreign, non-western background": (
                "First generation " "foreign, non-western"
            ),
            "Second generation foreign, Western background": (
                "Second generation " "foreign, Western"
            ),
            "Second generation foreign, non-western background": (
                "Second " "generation foreign, non-western"
            ),
            "Origin unknown or part of the information unknown (missing values)": pd.NA,
            "dutch background": "Dutch",
            "first generation foreign, western background": (
                "First generation " "foreign, Western"
            ),
            "first generation foreign, non-western background": (
                "First generation " "foreign, non-western"
            ),
            "second generation foreign, western background": (
                "Second generation " "foreign, Western"
            ),
            "second generation foreign, non-western background": (
                "Second generation " "foreign, non-western"
            ),
            "origin unknown or part of the information unknown (missing values)": pd.NA,
        },
    )

    return df
