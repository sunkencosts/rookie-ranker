import pandas as pd
import requests
from dotenv import load_dotenv
import os
from rapidfuzz import process

load_dotenv()


def load_data_from_api(year=2025):
    API_KEY = os.getenv("CFD_API_KEY")
    headers = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}
    url = f"https://api.collegefootballdata.com/stats/player/season?year={year}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Request failed with status code: {response.status_code}")

    json_data = response.json()
    df = pd.DataFrame(json_data)
    df.to_csv(f"data/college_stats_{year}.csv")


def transform_data(df: pd.DataFrame):
    categories = ["passing", "rushing", "receiving"]
    skill_positions = ["QB", "RB", "WR", "TE", "FB", "ATH"]
    offense_stats = df[df["position"].isin(skill_positions)]
    offense_stats = df[df["category"].isin(categories)]
    offense_stats["stat_name"] = (
        offense_stats["category"] + "_" + offense_stats["statType"]
    )

    df_wide = (
        offense_stats.pivot_table(
            index=["season", "playerId", "player", "position", "conference"],
            columns="stat_name",
            values="stat",
            aggfunc="sum",
        )
        .reset_index()
        .fillna(0)
    )
    # TODO Need to create another function that merges the data sets so that our training data has the nfl stats.
    # Need to be able to get the data by

    return df_wide


def merge_data(college_data, nfl_data):
    print("Merging data")

    college_names = college_data["player"].tolist()

    nfl_data["matched_name"] = nfl_data["pfr_player_name"].apply(
        lambda name: process.extractOne(name, college_names)[0]
    )

    merged_df = pd.merge(
        nfl_data,
        college_data,
        how="inner",
        left_on="matched_name",
        right_on="player",
    )

    print(f"Total matched: {len(merged_df)}")
    merged_df.to_csv("data/merged_nfl_college_2024.csv")
    return merged_df


def main():
    # 2025 college data - to predict on
    if not os.path.isfile("data/college_stats_2025.csv"):
        load_data_from_api("2025")
    # 2024 college data - to train on
    if not os.path.isfile("data/college_stats_2024.csv"):
        load_data_from_api("2024")

    df_college_2025 = pd.read_csv("data/college_stats_2025.csv")
    df_college_2024 = pd.read_csv("data/college_stats_2024.csv")

    df_college_2025 = transform_data(df_college_2025)
    df_college_2024 = transform_data(df_college_2024)

    # merge the college 2024 data with the NFL rookie 2025 data.
    df_nfl_2025 = pd.read_csv("data/draft_and_fantasy_data.csv")
    college_with_nfl_points = merge_data(df_college_2024, df_nfl_2025)
    print(college_with_nfl_points)


if __name__ == "__main__":
    main()
