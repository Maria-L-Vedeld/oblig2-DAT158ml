
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

wine_type = "white"#change to "white" id training white wine

# 1. Load red wine dataset
df = pd.read_csv(
    f"data/winequality-{wine_type}.csv",
    sep=";"
)

# 2. Separate input and target
X = df.drop(columns=["quality"])
y = df["quality"]

# 3. Create selected model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# 4. Train on all available data
model.fit(X, y)

# 5. Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    f"models/{wine_type}_model.joblib"
)

print("Final model trained and saved!")

