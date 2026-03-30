import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from .features import (
    prepare_features,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    NUMERIC_FEATURES_WITH_PICK,
    SKILL_POSITIONS,
    FBS_CONFERENCES,
)


def load_training_data(path):
    df = pd.read_csv(path)
    return prepare_features(df)


def build_model(numeric_features=NUMERIC_FEATURES):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )


def train(df, numeric_features=NUMERIC_FEATURES):
    x = df.drop(columns=["fantasy_points"])
    y = df["fantasy_points"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    model = build_model(numeric_features=numeric_features)
    model.fit(x_train, y_train)
    return model, x_test, y_test


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE={mse:.2f}  (lower is better — avg squared error)")
    print(f"R²={r2:.3f}  (1.0 is perfect)")
    return y_pred


def cross_validate(df, numeric_features=NUMERIC_FEATURES, cv=5):
    x = df.drop(columns=["fantasy_points"])
    y = df["fantasy_points"]
    model = build_model(numeric_features=numeric_features)
    scores = cross_val_score(model, x, y, cv=cv, scoring="r2")
    print(f"CV R² = {scores.mean():.3f} ± {scores.std():.3f}  (across {cv} folds)")


def load_model(path):
    return joblib.load(path)


def predict(model, df):
    # College-only data uses 'position'; merged training data uses 'position_x'
    if "position_x" not in df.columns and "position" in df.columns:
        df = df.rename(columns={"position": "position_x"})

    # Pre-filter to match what prepare_features will keep, so metadata stays aligned
    df_skill = df[df["position_x"].isin(SKILL_POSITIONS)].copy()
    if "conference" in df_skill.columns:
        df_skill = df_skill[df_skill["conference"].isin(FBS_CONFERENCES)].copy()
    player_names = df_skill["player"].tolist() if "player" in df_skill.columns else []
    conferences = df_skill["conference"].tolist() if "conference" in df_skill.columns else []

    df_features = prepare_features(df)
    predictions = model.predict(df_features)

    out = {
        "player": player_names,
        "position": df_features["position_x"].tolist(),
        "predicted_fantasy_points": predictions,
    }
    if conferences:
        out["conference"] = conferences
    return pd.DataFrame(out).sort_values("predicted_fantasy_points", ascending=False).reset_index(drop=True)


def explain_player(model, df, player_name):
    import shap

    if "position_x" not in df.columns and "position" in df.columns:
        df = df.rename(columns={"position": "position_x"})

    match = df[df["player"].str.lower() == player_name.lower()]
    if match.empty:
        print(f"Player '{player_name}' not found.")
        return

    df_features = prepare_features(df)
    player_features = prepare_features(match)

    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in model[:-1].get_feature_names_out()
    ]

    X_player = model[:-1].transform(player_features)
    if hasattr(X_player, "toarray"):
        X_player = X_player.toarray()

    explainer = shap.TreeExplainer(model.named_steps["regressor"])
    shap_values = explainer.shap_values(X_player)

    predicted = model.predict(player_features)[0]
    print(f"\n{player_name} — predicted: {predicted:.1f} fantasy pts\n")
    print(f"{'Feature':<35} {'SHAP':>8}")
    print("-" * 45)
    pairs = sorted(zip(feature_names, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)
    for name, val in pairs:
        if abs(val) < 0.01:
            continue
        direction = "↑" if val > 0 else "↓"
        print(f"{name:<35} {val:>+8.2f} {direction}")


def save_predictions(df_original, x_test, y_pred, path):
    df_out = df_original.loc[x_test.index].copy()
    df_out["y_pred"] = y_pred
    df_out.to_csv(path, index=False)
