import pandas as pd
import requests
from dotenv import load_dotenv
import os
from rapidfuzz import process

load_dotenv()

SKILL_POSITIONS = ["QB", "RB", "WR", "TE", "FB", "ATH"]
STAT_CATEGORIES = ["passing", "rushing", "receiving"]


def fetch_college_stats(year):
    API_KEY = os.getenv("CFD_API_KEY")
    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}
    url = f"https://api.collegefootballdata.com/stats/player/season?year={year}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code: {response.status_code}")
    return pd.DataFrame(response.json())


def load_college_data(year, force=False):
    path = f"data/raw/college_stats_{year}.csv"
    if not os.path.isfile(path) or force:
        os.makedirs("data/raw", exist_ok=True)
        df = fetch_college_stats(year)
        df.to_csv(path, index=False)
    return pd.read_csv(path)


def pivot_college_stats(df):
    offense_stats = df[
        df["position"].isin(SKILL_POSITIONS) & df["category"].isin(STAT_CATEGORIES)
    ].copy()
    offense_stats["stat_name"] = (
        offense_stats["category"] + "_" + offense_stats["statType"]
    )

    return (
        offense_stats.pivot_table(
            index=["season", "playerId", "player", "position", "conference"],
            columns="stat_name",
            values="stat",
            aggfunc="sum",
        )
        .reset_index()
        .fillna(0)
    )


def get_multi_season_players(years, min_seasons=3):
    """Return set of playerIds who appear in at least min_seasons of the given years."""
    from collections import Counter
    counts = Counter()
    for year in years:
        df = load_college_data(year)
        counts.update(df["playerId"].unique())
    return {pid for pid, count in counts.items() if count >= min_seasons}


def merge_college_and_nfl(college_data, nfl_data):
    college_names = college_data["player"].tolist()
    nfl_data = nfl_data.copy()
    nfl_data["matched_name"] = nfl_data["pfr_player_name"].apply(
        lambda name: process.extractOne(name, college_names)[0]
    )
    merged = pd.merge(
        nfl_data,
        college_data,
        how="inner",
        left_on="matched_name",
        right_on="player",
    )
    merged = merged.drop(columns=["matched_name"])
    print(f"Total matched: {len(merged)}")
    return merged
