import pandas as pd

# Columns that are join artifacts or otherwise irrelevant as model features
ARTIFACT_COLUMNS = [
    "season_x", "round", "gsis_id", "player_id", "playerId",
    "player", "position_y", "season_y", "pfr_player_name",
]

# Non-skill positions that can appear after the college/NFL merge
NON_SKILL_POSITIONS = ["DE", "OT", "DT", "DL", "SAF", "DB", "LS", "CB", "OL"]

CATEGORICAL_FEATURES = ["position_x", "conference"]
NUMERIC_FEATURES = [
    "pick",
    "passing_YDS", "passing_TD",
    "rushing_YDS", "rushing_TD",
    "receiving_YDS", "receiving_TD",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a merged college/NFL dataframe into model-ready features.
    Used by both training and inference so the same cleanup is never duplicated.
    """
    df = df.drop(columns=ARTIFACT_COLUMNS, errors="ignore")
    df = df[~df["position_x"].isin(NON_SKILL_POSITIONS)]
    return df.reset_index(drop=True)
