import argparse
import os
import joblib
import pandas as pd

from rookie_ranker.nfl_data import load_nfl_data
from rookie_ranker.college_data import load_college_data, pivot_college_stats, merge_college_and_nfl
from rookie_ranker.model import load_training_data, train, evaluate, save_predictions

NFL_YEAR = 2025
COLLEGE_YEAR = NFL_YEAR - 1
NFL_PATH = f"data/raw/nfl_rookies_{NFL_YEAR}.csv"
TRAINING_DATA_PATH = f"data/processed/training_data_{COLLEGE_YEAR}.csv"
PREDICTIONS_PATH = "data/output/test_predictions.csv"
MODEL_PATH = "models/fantasy_model.pkl"


def run_pipeline(force=False):
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # --- NFL data ---
    if not os.path.isfile(NFL_PATH) or force:
        print(f"Fetching NFL {NFL_YEAR} data...")
        df_nfl = load_nfl_data(NFL_YEAR)
        df_nfl.to_csv(NFL_PATH, index=False)
        print(f"  {len(df_nfl)} rookies saved.")
    else:
        df_nfl = pd.read_csv(NFL_PATH)
        print(f"NFL data loaded from cache ({len(df_nfl)} rows).")

    # --- College data + merge ---
    if not os.path.isfile(TRAINING_DATA_PATH) or force:
        print(f"Fetching/transforming college {COLLEGE_YEAR} data...")
        df_college = pivot_college_stats(load_college_data(COLLEGE_YEAR, force=force))
        df_merged = merge_college_and_nfl(df_college, df_nfl)
        df_merged.to_csv(TRAINING_DATA_PATH, index=False)
        print(f"  {len(df_merged)} matched players saved.")
    else:
        print(f"Training data loaded from cache ({TRAINING_DATA_PATH}).")

    # --- Train model ---
    print("Training model...")
    df = load_training_data(TRAINING_DATA_PATH)
    model, x_test, y_test = train(df)
    y_pred = evaluate(model, x_test, y_test)
    save_predictions(df, x_test, y_pred, PREDICTIONS_PATH)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Rookie Ranker pipeline")
    parser.add_argument(
        "--force", action="store_true", help="Force reload all data from APIs"
    )
    args = parser.parse_args()
    run_pipeline(force=args.force)


if __name__ == "__main__":
    main()
