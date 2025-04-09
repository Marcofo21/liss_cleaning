import pandas as pd

pd.set_option("future.no_silent_downcasting", True)


def clean_dataset(raw):
    df = pd.DataFrame()
    options_ambiguous = {
        1: "e0",
        2: "e1",
        3: "e2",
        4: "e3",
        5: "e1c",
        6: "e2c",
        7: "e3c",
    }

    df["personal_id"] = raw["personal_id"]
    for option in options_ambiguous.values():
        matching_columns = [col for col in raw.columns if option in col]
        df[f"mp_{option}"] = raw[matching_columns].apply(
            lambda x, opt=option: _get_interval(x, opt), axis=1
        )
    df["wave"] = raw["wave"]
    df["survey_completion"] = raw["end_time"]
    df = df.groupby(["personal_id", "wave"]).filter(_check_answered_all_questions)
    filtered_df = df.groupby("personal_id").filter(_check_eligible)
    return filtered_df


def _check_answered_all_questions(row):
    """Return True if all questions in a row are answered, False otherwise."""
    question_cols = ["mp_e0", "mp_e1", "mp_e2", "mp_e3", "mp_e1c", "mp_e2c", "mp_e3c"]
    return row[question_cols].notna().all(axis=1).all()


def _check_eligible(individual_data):
    """Return False if there is less than 2 waves of data, True otherwise."""
    individual_data = individual_data.drop(["personal_id", "wave"], axis=1)

    return not individual_data.dropna().shape[0] < 2


def _get_interval(rows, option):  # noqa: C901, PLR0912
    if rows.isna().all():
        return pd.NA
    matching_prob_interval = (0, 0)
    if rows[f"choice_aex_{option}_vs_50"] == "AEX":
        if rows[f"choice_aex_{option}_vs_90"] == "AEX":
            if rows[f"choice_aex_{option}_vs_95"] == "AEX":
                if rows[f"choice_aex_{option}_vs_99"] == "AEX":
                    matching_prob_interval = (0.99, 1)
                if rows[f"choice_aex_{option}_vs_99"] == "Lottery":
                    matching_prob_interval = (0.95, 0.99)
            if rows[f"choice_aex_{option}_vs_95"] == "Lottery":
                matching_prob_interval = (0.9, 0.95)
        if rows[f"choice_aex_{option}_vs_90"] == "Lottery":
            if rows[f"choice_aex_{option}_vs_70"] == "AEX":
                if rows[f"choice_aex_{option}_vs_80"] == "AEX":
                    matching_prob_interval = (0.8, 0.9)
                if rows[f"choice_aex_{option}_vs_80"] == "Lottery":
                    matching_prob_interval = (0.7, 0.8)
            if rows[f"choice_aex_{option}_vs_70"] == "Lottery":
                if rows[f"choice_aex_{option}_vs_60"] == "AEX":
                    matching_prob_interval = (0.6, 0.7)
                if rows[f"choice_aex_{option}_vs_60"] == "Lottery":
                    matching_prob_interval = (0.5, 0.6)
    if rows[f"choice_aex_{option}_vs_50"] == "Lottery":
        if rows[f"choice_aex_{option}_vs_10"] == "AEX":
            if rows[f"choice_aex_{option}_vs_30"] == "AEX":
                if rows[f"choice_aex_{option}_vs_40"] == "AEX":
                    matching_prob_interval = (0.4, 0.5)
                if rows[f"choice_aex_{option}_vs_40"] == "Lottery":
                    matching_prob_interval = (0.3, 0.4)
            if rows[f"choice_aex_{option}_vs_30"] == "Lottery":
                if rows[f"choice_aex_{option}_vs_20"] == "AEX":
                    matching_prob_interval = (0.2, 0.3)
                if rows[f"choice_aex_{option}_vs_20"] == "Lottery":
                    matching_prob_interval = (0.1, 0.2)
        if rows[f"choice_aex_{option}_vs_10"] == "Lottery":
            if rows[f"choice_aex_{option}_vs_5"] == "AEX":
                matching_prob_interval = (0.05, 0.1)
            if rows[f"choice_aex_{option}_vs_5"] == "Lottery":
                if rows[f"choice_aex_{option}_vs_1"] == "AEX":
                    matching_prob_interval = (0.01, 0.05)
                if rows[f"choice_aex_{option}_vs_1"] == "Lottery":
                    matching_prob_interval = (0, 0.01)
    return matching_prob_interval
