import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from .features import prepare_features, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def load_training_data(path):
    df = pd.read_csv(path)
    return prepare_features(df)


def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
    ])


def train(df):
    x = df.drop(columns=["fantasy_points"])
    y = df["fantasy_points"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = build_model()
    model.fit(x_train, y_train)
    return model, x_test, y_test


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE={mse:.2f}  (lower is better — avg squared error)")
    print(f"R²={r2:.3f}  (1.0 is perfect)")
    return y_pred


def save_predictions(df_original, x_test, y_pred, path):
    df_out = df_original.loc[x_test.index].copy()
    df_out["y_pred"] = y_pred
    df_out.to_csv(path, index=False)
