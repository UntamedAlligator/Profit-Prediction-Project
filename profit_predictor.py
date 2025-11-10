# profit_predictor.py

# Import required libraries
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("50_Startups.csv")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Visualizations
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Feature selection
X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
y = df['Profit']

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=0),
    "Random Forest": RandomForestRegressor(random_state=0)
}

print("\nModel Scores:")
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}: R^2 Score = {score:.2f}")

# Evaluate best model
best_model = models["Random Forest"]
y_pred = best_model.predict(X_test)

print("\nEvaluation Metrics:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred) ** 0.5)  # Manual RMSE for compatibility
print("R² Score:", r2_score(y_test, y_pred))

# Save model to disk
joblib.dump(best_model, "profit_predictor.pkl")
print("\nModel saved as 'profit_predictor.pkl'")

