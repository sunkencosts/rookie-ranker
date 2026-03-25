import nflreadpy as nfl
import pandas as pd


def get_player_stats(year):
    player_stats = nfl.load_player_stats(year)
    player_stats = player_stats.to_pandas()

    return player_stats


def get_rookie_fantasy_points(df):
    filtered_player_stats = df.filter(
        items=[
            "player_id",
            "player_display_name",
            "season",
            "season_type",
            "position",
            "fantasy_points",
        ]
    )
    fantasy_points = pd.DataFrame({"player_id": [], "fantasy_points": []})
    seen = set()
    for player in filtered_player_stats["player_id"]:
        if player not in seen:
            points = (
                filtered_player_stats.loc[
                    filtered_player_stats["player_id"] == player,
                    filtered_player_stats.columns.str.contains("fantasy_points"),
                ]
                .sum()
                .sum()
            )
            fantasy_points.loc[len(fantasy_points)] = {
                "player_id": player,
                "fantasy_points": points,
            }
            seen.add(player)

    fantasy_points.to_csv("data/fantasy_points.csv")
    return fantasy_points


def get_draft_picks(year):
    draft_picks = nfl.load_draft_picks(year)
    draft_picks = draft_picks.to_pandas()
    return draft_picks


def filter_draft_picks(df):
    filtered_draft_picks = df.filter(
        items=["season", "round", "pick", "position", "pfr_player_name", "gsis_id"]
    )
    filtered_draft_picks.to_csv("data/draft_picks.csv")
    return filtered_draft_picks


def merge_dfs(filtered_draft_picks, fantasy_points):
    merged_df = pd.merge(
        filtered_draft_picks,
        fantasy_points,
        how="inner",
        left_on="gsis_id",
        right_on="player_id",
    )
    merged_df.to_csv("data/draft_and_fantasy_data.csv")
    return merged_df


def main():
    player_stats = get_player_stats(2025)
    fantasy_points = get_rookie_fantasy_points(player_stats)
    draft_info = get_draft_picks(2025)
    draft_info = filter_draft_picks(draft_info)
    merge_dfs(fantasy_points=fantasy_points, filtered_draft_picks=draft_info)
    return "Done! Rookie Season 2025 saved to data/draft_and_fantasy_data.csv"


if __name__ == "__main__":
    main()
