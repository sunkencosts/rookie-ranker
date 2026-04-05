import argparse
import csv
import os
import joblib
import pandas as pd
from datetime import datetime

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
# With-pick uses all years — draft position is a stable signal across eras
# No-pick uses recent years only — college stat patterns have shifted with rule changes
TRAINING_YEARS_ALL = [
    (2014, 2015),
    (2015, 2016),
    (2016, 2017),
    (2017, 2018),
    (2018, 2019),
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025),
]
TRAINING_YEARS_RECENT = [y for y in TRAINING_YEARS_ALL if y[0] >= 2020]

COMBINED_TRAINING_PATH = "data/processed/training_data_combined.csv"
PREDICTIONS_PATH = "data/output/test_predictions.csv"
MODEL_PATH_NO_PICK = "models/model_no_pick.pkl"
MODEL_PATH_WITH_PICK = "models/model_with_pick.pkl"
MODEL_LOG_PATH = "data/model_log.csv"

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


def log_run(n_rows, cv_no_pick, cv_std_no_pick, cv_with_pick, cv_std_with_pick, note=""):
    """Append a training run entry to the model log."""
    os.makedirs("data", exist_ok=True)
    write_header = not os.path.isfile(MODEL_LOG_PATH)
    with open(MODEL_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "note", "years_no_pick", "years_with_pick", "n_rows_no_pick",
                "cv_r2_no_pick", "cv_std_no_pick",
                "cv_r2_with_pick", "cv_std_with_pick",
            ])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            note,
            len(TRAINING_YEARS_RECENT),
            len(TRAINING_YEARS_ALL),
            n_rows,
            f"{cv_no_pick:.3f}",
            f"{cv_std_no_pick:.3f}",
            f"{cv_with_pick:.3f}",
            f"{cv_std_with_pick:.3f}",
        ])


def run_pipeline(force=False, note=""):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # --- Fetch all year pairs (full history) ---
    for college_year, nfl_year in TRAINING_YEARS_ALL:
        fetch_year(college_year, nfl_year, force=force)

    # No-pick: recent years only (college stat patterns shift with rule changes)
    recent_frames = [pd.read_csv(f"data/processed/training_data_{cy}.csv") for cy, _ in TRAINING_YEARS_RECENT]
    df_recent = pd.concat(recent_frames, ignore_index=True)
    df_recent.to_csv(COMBINED_TRAINING_PATH.replace(".csv", "_recent.csv"), index=False)

    # With-pick: all years (draft position is stable across eras)
    all_frames = [pd.read_csv(f"data/processed/training_data_{cy}.csv") for cy, _ in TRAINING_YEARS_ALL]
    df_all = pd.concat(all_frames, ignore_index=True)
    df_all.to_csv(COMBINED_TRAINING_PATH, index=False)

    print(f"\nNo-pick training: {len(df_recent)} rows ({len(TRAINING_YEARS_RECENT)} years)")
    print(f"With-pick training: {len(df_all)} rows ({len(TRAINING_YEARS_ALL)} years)")

    df_recent_prep = load_training_data(COMBINED_TRAINING_PATH.replace(".csv", "_recent.csv"))
    df_all_prep = load_training_data(COMBINED_TRAINING_PATH)

    print("\nEvaluating model (no pick)...")
    cv_no_pick, cv_std_no_pick = cross_validate(df_recent_prep, numeric_features=NUMERIC_FEATURES)
    model_no_pick, x_test, y_test = train(df_recent_prep, numeric_features=NUMERIC_FEATURES)
    y_pred = evaluate(model_no_pick, x_test, y_test)
    save_predictions(df_recent_prep, x_test, y_pred, PREDICTIONS_PATH)
    joblib.dump(model_no_pick, MODEL_PATH_NO_PICK)
    print(f"Model saved to {MODEL_PATH_NO_PICK}")

    print("\nEvaluating model (with pick)...")
    cv_with_pick, cv_std_with_pick = cross_validate(df_all_prep, numeric_features=NUMERIC_FEATURES_WITH_PICK)
    model_with_pick, x_test, y_test = train(df_all_prep, numeric_features=NUMERIC_FEATURES_WITH_PICK)
    evaluate(model_with_pick, x_test, y_test)
    joblib.dump(model_with_pick, MODEL_PATH_WITH_PICK)
    print(f"Model saved to {MODEL_PATH_WITH_PICK}")

    log_run(len(df_recent_prep), cv_no_pick, cv_std_no_pick, cv_with_pick, cv_std_with_pick, note=note)
    print(f"\nRun logged to {MODEL_LOG_PATH}")


def print_position_group(group, label):
    if group.empty:
        return
    print(f"\n  {label}")
    print(f"  {'#':<4} {'Player':<26} {'Conf':<20} {'Pts':>6}  {'Gap':>6}")
    print(f"  {'-'*64}")
    prev_pts = None
    for rank, (_, row) in enumerate(group.iterrows(), 1):
        gap = f"-{prev_pts - row['predicted_fantasy_points']:.1f}" if prev_pts is not None else ""
        print(f"  {rank:<4} {row['player']:<26} {row['conference']:<20} {row['predicted_fantasy_points']:>6.1f}  {gap:>6}")
        prev_pts = row['predicted_fantasy_points']


def print_rankings(rankings):
    top_per_pos = {"QB": 5, "RB": 8, "WR": 10, "TE": 5, "FB": 2, "ATH": 2}
    has_mixed = "model" in rankings.columns and rankings["model"].nunique() > 1

    print(f"\n{PREDICT_YEAR} Rookie Rankings\n{'='*60}")
    for pos, n in top_per_pos.items():
        pos_group = rankings[rankings["position"] == pos]
        if pos_group.empty:
            continue

        if has_mixed:
            with_pick = pos_group[pos_group["model"] == "with_pick"].head(n).reset_index(drop=True)
            no_pick = pos_group[pos_group["model"] == "no_pick"].head(n).reset_index(drop=True)
            if not with_pick.empty:
                print_position_group(with_pick, f"{pos}  [draft-projected, with-pick model]")
            if not no_pick.empty:
                print_position_group(no_pick, f"{pos}  [no projection, no-pick model]")
        else:
            print_position_group(pos_group.head(n).reset_index(drop=True), pos)


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

    draft_proj_path = f"data/raw/draft_projections_{PREDICT_YEAR}.csv"
    if os.path.isfile(draft_proj_path):
        df_proj = pd.read_csv(draft_proj_path)[["player", "pick"]]
        df_merged = df_prospects.merge(df_proj, on="player", how="left")
        has_pick = df_merged["pick"].notna()
        n_with_pick = has_pick.sum()

        print(f"\n{n_with_pick} prospects have draft projections — using with-pick model for those.")
        model_no_pick = load_model(MODEL_PATH_NO_PICK)
        model_with_pick = load_model(MODEL_PATH_WITH_PICK)

        r1 = predict(model_with_pick, df_merged[has_pick].copy())
        r1["model"] = "with_pick"
        r2 = predict(model_no_pick, df_merged[~has_pick].copy())
        r2["model"] = "no_pick"
        rankings = pd.concat([r1, r2], ignore_index=True)
    else:
        model_no_pick = load_model(MODEL_PATH_NO_PICK)
        rankings = predict(model_no_pick, df_prospects)
        rankings["model"] = "no_pick"

    rankings["position_rank"] = (
        rankings.groupby("position")["predicted_fantasy_points"]
        .rank(ascending=False)
        .astype(int)
    )
    rankings.to_csv(RANKINGS_PATH, index=False)
    print_rankings(rankings)


def main():
    parser = argparse.ArgumentParser(description="Rookie Ranker pipeline")
    parser.add_argument("--force", action="store_true", help="Force reload all data from APIs")
    parser.add_argument("--predict", action="store_true", help="Predict points for upcoming rookie class")
    parser.add_argument("--explain", type=str, metavar="PLAYER", help="Explain prediction for a specific player")
    parser.add_argument("--note", type=str, default="", help="Note to attach to this training run in the log")
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
        run_pipeline(force=args.force, note=args.note)


if __name__ == "__main__":
    main()
