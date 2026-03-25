# MVP model test
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# load data - Ben switched it to use new df
df = pd.read_csv("data/merged_nfl_college_2024.csv")
df_copy = df.copy()

# remove un-needed cols
columns_to_drop = [
    "Unnamed: 0",
    "season_x",
    "round",
    "gsis_id",
    "player_id",
    "playerId",
    "matched_name",
    "player",
    "position_y",
    "season_y",
    "pfr_player_name",
]
df.drop(columns=columns_to_drop, inplace=True, errors="ignore")
print("Columns after all drops", df.columns.to_list())
print("DATA FORM AFTER DROP")
print(df)

# target
target = "fantasy_points"
x = df.drop(columns=[target])
y = df[target]

# categorical encoding
cat_cols = [
    "position_x",
    "conference",
]
num_cols = [
    "pick",
    "passing_ATT",
    "passing_COMPLETIONS",
    "passing_INT",
    "passing_PCT",
    "passing_TD",
    "passing_YDS",
    "passing_YPA",
    "receiving_LONG",
    "receiving_REC",
    "receiving_TD",
    "receiving_YDS",
    "receiving_YPR",
    "rushing_CAR",
    "rushing_LONG",
    "rushing_TD",
    "rushing_YDS",
    "rushing_YPC",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ]
)

# split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# RF
model_RF = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
    ]
)

model_RF.fit(x_train, y_train)

df_test = df_copy.loc[x_test.index].copy()
y_pred = model_RF.predict(x_test)

df_test["y_pred"] = y_pred

# evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE = {mse}")
print(f"r2 = {r2}")

df_test.to_csv("data/final_model_pred.csv")
