
from prepareData import load_and_split_data

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load the same training and test data
X_train, X_test, y_train, y_test = load_and_split_data("white")  #change to red if ypu want a pred for red

# Create model
model = GradientBoostingRegressor(
    random_state=42
)

# Cross-validation on training data
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error"
)

cv_mae = -cv_scores.mean()

# Train model
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nGRADIENT BOOSTING")
print("-----------------")
print(f"Cross-validation MAE: {cv_mae:.3f}")
print(f"Test MAE: {mae:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test R2: {r2:.3f}")
