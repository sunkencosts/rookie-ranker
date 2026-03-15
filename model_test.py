# MVP model test
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# load data
df = pd.read_csv("draft_and_fantasy_data.csv")
df_copy = df.copy()

# remove un-needed cols
df.drop(columns=["player_id", "season", "gsis_id", "pfr_player_name"], inplace=True)

# target
target = "fantasy_points"
x = df.drop(columns=[target])
y = df[target]

# categorical encoding
cat_cols = ["position"]
num_cols = ["round", "pick"]

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

df_test.to_csv("final_model_pred.csv")
