import nflreadpy as nfl
import pandas as pd


def load_player_stats(year):
    return nfl.load_player_stats(year).to_pandas()


def aggregate_fantasy_points(df):
    return (
        df[["player_id", "fantasy_points"]]
        .groupby("player_id", as_index=False)["fantasy_points"]
        .sum()
    )


def load_draft_picks(year):
    return nfl.load_draft_picks(year).to_pandas()


def select_draft_columns(df):
    return df[["season", "round", "pick", "position", "pfr_player_name", "gsis_id"]]


def merge_draft_and_fantasy(draft_picks, fantasy_points):
    return pd.merge(
        draft_picks,
        fantasy_points,
        how="inner",
        left_on="gsis_id",
        right_on="player_id",
    )


def load_nfl_data(year):
    player_stats = load_player_stats(year)
    fantasy_points = aggregate_fantasy_points(player_stats)
    draft_picks = select_draft_columns(load_draft_picks(year))
    return merge_draft_and_fantasy(draft_picks, fantasy_points)
