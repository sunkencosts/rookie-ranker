import argparse
import os
import joblib
import pandas as pd

from rookie_ranker.nfl_data import load_nfl_data
from rookie_ranker.college_data import (
    load_college_data,
    pivot_college_stats,
    merge_college_and_nfl,
    get_multi_season_players,
)
from rookie_ranker.model import load_training_data, train, evaluate, save_predictions, cross_validate, load_model, predict, explain_player
from rookie_ranker.features import NUMERIC_FEATURES, NUMERIC_FEATURES_WITH_PICK, filter_prospects

# Each tuple is (college_year, nfl_rookie_year)
TRAINING_YEARS = [
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]

COMBINED_TRAINING_PATH = "data/processed/training_data_combined.csv"
PREDICTIONS_PATH = "data/output/test_predictions.csv"
MODEL_PATH_NO_PICK = "models/model_no_pick.pkl"
MODEL_PATH_WITH_PICK = "models/model_with_pick.pkl"

PREDICT_YEAR = 2026
PREDICT_COLLEGE_YEAR = PREDICT_YEAR - 1
PREDICT_COLLEGE_PATH = f"data/processed/predict_data_{PREDICT_YEAR}.csv"
RANKINGS_PATH = f"data/output/rookie_rankings_{PREDICT_YEAR}.csv"


def fetch_year(college_year, nfl_year, force=False):
    """Fetch, merge, and cache one year of college + NFL data. Returns a DataFrame."""
    nfl_path = f"data/raw/nfl_rookies_{nfl_year}.csv"
    merged_path = f"data/processed/training_data_{college_year}.csv"

    if not os.path.isfile(nfl_path) or force:
        print(f"  Fetching NFL {nfl_year} data...")
        df_nfl = load_nfl_data(nfl_year)
        df_nfl.to_csv(nfl_path, index=False)
    else:
        df_nfl = pd.read_csv(nfl_path)

    if not os.path.isfile(merged_path) or force:
        print(f"  Merging college {college_year} → NFL {nfl_year}...")
        df_college = pivot_college_stats(load_college_data(college_year, force=force))
        df_merged = merge_college_and_nfl(df_college, df_nfl)
        df_merged.to_csv(merged_path, index=False)
        print(f"  {len(df_merged)} players matched.")
    else:
        df_merged = pd.read_csv(merged_path)
        print(f"  {college_year}→{nfl_year}: loaded from cache ({len(df_merged)} rows).")

    return df_merged


def run_pipeline(force=False):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # --- Fetch all year pairs ---
    frames = []
    for college_year, nfl_year in TRAINING_YEARS:
        df = fetch_year(college_year, nfl_year, force=force)
        frames.append(df)

    df_combined = pd.concat(frames, ignore_index=True)
    df_combined.to_csv(COMBINED_TRAINING_PATH, index=False)
    print(f"\nCombined training set: {len(df_combined)} rows across {len(TRAINING_YEARS)} years.")

    # --- Train models ---
    df = load_training_data(COMBINED_TRAINING_PATH)

    print("\nEvaluating model (no pick)...")
    cross_validate(df, numeric_features=NUMERIC_FEATURES)
    model_no_pick, x_test, y_test = train(df, numeric_features=NUMERIC_FEATURES)
    y_pred = evaluate(model_no_pick, x_test, y_test)
    save_predictions(df, x_test, y_pred, PREDICTIONS_PATH)
    joblib.dump(model_no_pick, MODEL_PATH_NO_PICK)
    print(f"Model saved to {MODEL_PATH_NO_PICK}")

    print("\nEvaluating model (with pick)...")
    cross_validate(df, numeric_features=NUMERIC_FEATURES_WITH_PICK)
    model_with_pick, x_test, y_test = train(df, numeric_features=NUMERIC_FEATURES_WITH_PICK)
    evaluate(model_with_pick, x_test, y_test)
    joblib.dump(model_with_pick, MODEL_PATH_WITH_PICK)
    print(f"Model saved to {MODEL_PATH_WITH_PICK}")


def run_predict(force=False):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)

    if not os.path.isfile(PREDICT_COLLEGE_PATH) or force:
        print(f"Fetching college {PREDICT_COLLEGE_YEAR} data...")
        df_college = pivot_college_stats(load_college_data(PREDICT_COLLEGE_YEAR, force=force))
        df_college.to_csv(PREDICT_COLLEGE_PATH, index=False)
        print(f"  {len(df_college)} players saved.")
    else:
        df_college = pd.read_csv(PREDICT_COLLEGE_PATH)
        print(f"College data loaded from cache ({len(df_college)} rows).")

    df_prospects = filter_prospects(df_college)
    print(f"Filtered to {len(df_prospects)} prospects by stat thresholds.")

    returning = get_multi_season_players([2022, 2023, 2024, 2025], min_seasons=3)
    df_prospects = df_prospects[df_prospects["playerId"].isin(returning)]
    print(f"Filtered to {len(df_prospects)} prospects with 3+ college seasons.")

    model = load_model(MODEL_PATH_NO_PICK)
    rankings = predict(model, df_prospects)
    rankings["position_rank"] = (
        rankings.groupby("position")["predicted_fantasy_points"]
        .rank(ascending=False)
        .astype(int)
    )
    rankings.to_csv(RANKINGS_PATH, index=False)

    print(f"\n{PREDICT_YEAR} Rookie Rankings by Position\n")
    top_per_pos = {"QB": 5, "RB": 8, "WR": 10, "TE": 5, "FB": 2, "ATH": 2}
    for pos, n in top_per_pos.items():
        group = rankings[rankings["position"] == pos].head(n)
        if group.empty:
            continue
        print(f"--- {pos} ---")
        print(group[["position_rank", "player", "conference", "predicted_fantasy_points"]].to_string(index=False))
        print()


def main():
    parser = argparse.ArgumentParser(description="Rookie Ranker pipeline")
    parser.add_argument(
        "--force", action="store_true", help="Force reload all data from APIs"
    )
    parser.add_argument("--predict", action="store_true", help="Predict points for upcoming rookie class")
    parser.add_argument("--explain", type=str, metavar="PLAYER", help="Explain prediction for a specific player")
    args = parser.parse_args()

    if args.explain:
        df_college = pd.read_csv(PREDICT_COLLEGE_PATH)
        df_prospects = filter_prospects(df_college)
        returning = get_multi_season_players([2022, 2023, 2024, 2025], min_seasons=3)
        df_prospects = df_prospects[df_prospects["playerId"].isin(returning)]
        model = load_model(MODEL_PATH_NO_PICK)
        explain_player(model, df_prospects, args.explain)
    elif args.predict:
        run_predict(force=args.force)
    else:
        run_pipeline(force=args.force)


if __name__ == "__main__":
    main()
