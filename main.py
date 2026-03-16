import nflreadpy as nfl
import pandas as pd


def main():
    player_stats = nfl.load_players()
    df = player_stats.to_pandas()
    print(df.columns.to_list())
    df = df.filter(items=['display_name','gsis_id', 'rookie_season','draft_round','draft_pick','draft_year'])
    players_2024 = df[df["draft_year"]==2025]
    print(players_2024)

    players_2024.to_csv("df_output.csv")
   


if __name__ == "__main__":
    main()
