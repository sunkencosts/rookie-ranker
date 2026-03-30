import pandas as pd

# Columns that are join artifacts or otherwise irrelevant as model features
ARTIFACT_COLUMNS = [
    "season_x",
    "round",
    "gsis_id",
    "player_id",
    "playerId",
    "player",
    "position_y",
    "season_y",
    "pfr_player_name",
    "conference",
]

# Allowlist — only these positions are kept. Anything else is filtered out.
SKILL_POSITIONS = ["QB", "RB", "WR", "TE", "FB", "ATH"]

# FBS conferences only — excludes FCS players whose stats don't translate
FBS_CONFERENCES = [
    "ACC",
    "Big Ten",
    "Big 12",
    "SEC",
    "Pac-12",
    "American Athletic",
    "Conference USA",
    "Mid-American",
    "Mountain West",
    "Sun Belt",
    "FBS Independents",
]

CATEGORICAL_FEATURES = ["position_x"]
NUMERIC_FEATURES = [
    # raw stats
    "passing_YDS",
    "passing_TD",
    "passing_YPA",
    "rushing_YDS",
    "rushing_TD",
    "rushing_CAR",
    "receiving_YDS",
    "receiving_TD",
    "receiving_REC",
    # derived
    "total_touchdowns",
    "yards_from_scrimmage",
]
NUMERIC_FEATURES_WITH_PICK = ["pick"] + NUMERIC_FEATURES


def filter_prospects(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter college players to plausible draft prospects using position-specific
    stat thresholds and FBS conference membership.
    """
    pos = "position_x" if "position_x" in df.columns else "position"
    conf = "conference"

    masks = [
        (df[pos] == "QB")  & (df["passing_YDS"] >= 1500),
        (df[pos] == "RB")  & ((df["rushing_YDS"] >= 500) | (df["receiving_YDS"] >= 300)),
        (df[pos] == "WR")  & (df["receiving_YDS"] >= 500),
        (df[pos] == "TE")  & (df["receiving_YDS"] >= 250),
        (df[pos] == "FB")  & ((df["rushing_YDS"] >= 200) | (df["receiving_YDS"] >= 150)),
        (df[pos] == "ATH") & ((df["rushing_YDS"] >= 300) | (df["receiving_YDS"] >= 200)),
    ]
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined | mask

    return df[combined & df[conf].isin(FBS_CONFERENCES)].copy()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a merged college/NFL dataframe into model-ready features.
    Used by both training and inference so the same cleanup is never duplicated.
    """
    # Filter to FBS only before dropping conference
    if "conference" in df.columns:
        df = df[df["conference"].isin(FBS_CONFERENCES)].copy()

    df = df.drop(columns=ARTIFACT_COLUMNS, errors="ignore")
    df = df[df["position_x"].isin(SKILL_POSITIONS)].copy()

    # Zero out all passing stats for non-QBs or players with < 5 attempts
    if "passing_ATT" in df.columns:
        non_passer = df["passing_ATT"] < 5
        for col in ["passing_YDS", "passing_TD", "passing_YPA"]:
            if col in df.columns:
                df.loc[non_passer, col] = 0

    df["total_touchdowns"] = df["passing_TD"] + df["rushing_TD"] + df["receiving_TD"]
    df["yards_from_scrimmage"] = df["rushing_YDS"] + df["receiving_YDS"]
    return df.reset_index(drop=True)
