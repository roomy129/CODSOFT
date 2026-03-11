# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace with your file path)
df = pd.read_csv("D:\\IMDb Movies India.csv")

# Quick look at data
print(df.head())
print(df.info())

# -----------------------------
# Data Preprocessing
# -----------------------------

# Handle missing values
df = df.dropna(subset=['rating'])   # drop rows without rating
df.fillna({'budget': df['budget'].median(),
           'runtime': df['runtime'].median()}, inplace=True)

# Encode categorical features
label_enc = LabelEncoder()
for col in ['genre', 'director', 'production_company']:
    if col in df.columns:
        df[col] = label_enc.fit_transform(df[col].astype(str))

# Feature selection
features = ['budget', 'runtime', 'genre', 'director', 'production_company', 'year']
X = df[features]
y = df['rating']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model Training
# -----------------------------

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# XGBoost
xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"{name} Performance:")
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R² Score:", r2_score(y_true, y_pred))
    print("-"*40)

evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("XGBoost", y_test, xgb_preds)

# -----------------------------
# Feature Importance
# -----------------------------
import matplotlib.pyplot as plt

feat_importances = pd.Series(rf_model.feature_importances_, index=features)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance - Random Forest")
plt.show()