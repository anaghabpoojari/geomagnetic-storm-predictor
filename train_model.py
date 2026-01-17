# Step 0: Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load dataset
file_path = r"C:\Users\Dell\Downloads\geomagnetic_storms.csv"
df = pd.read_csv(file_path)

# Step 2: Drop empty columns
df = df.drop(columns=['peak_time', 'end_time', 'source_location', 'active_region'])

# Step 3: Convert datetime columns to datetime
df['begin_time'] = pd.to_datetime(df['begin_time'])
df['observed_time'] = pd.to_datetime(df['observed_time'])

# Step 4: Feature engineering
# Storm duration in hours
df['duration_hours'] = (df['observed_time'] - df['begin_time']).dt.total_seconds() / 3600.0

# Time-based features
df['hour_of_day'] = df['begin_time'].dt.hour
df['day_of_year'] = df['begin_time'].dt.dayofyear

# Step 5: One-hot encode categorical variables
categorical_features = ['event_type', 'class_type']
ohe = OneHotEncoder(
    sparse_output=False,
    drop='first',
    handle_unknown='ignore'
)
cat_encoded = ohe.fit_transform(df[categorical_features])
cat_feature_names = ohe.get_feature_names_out(categorical_features)
df_encoded = pd.DataFrame(cat_encoded, columns=cat_feature_names)

# Step 6: Combine features
df = pd.concat([df, df_encoded], axis=1)

# Step 7: Add lag feature for previous kp_index
# Sort by observed_time first
df = df.sort_values('observed_time').reset_index(drop=True)
df['kp_index_lag1'] = df['kp_index'].shift(1)  # previous row kp_index
df['kp_index_lag1'].fillna(df['kp_index'].mean(), inplace=True)  # fill first row

# Step 8: Prepare final features and target
feature_columns = ['duration_hours', 'hour_of_day', 'day_of_year', 'kp_index_lag1'] + list(cat_feature_names)
X = df[feature_columns].values
y = df['kp_index'].values

# Step 9: Scale features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Step 10: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 11: Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Step 12: Predict and evaluate
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse}")
print(f"Test R^2 Score: {r2}")

# Step 13: Save model and scaler
joblib.dump(rf_model, "geomagnetic_rf_model_v2.pkl")
joblib.dump(scaler_X, "geomagnetic_scaler_v2.pkl")
joblib.dump(ohe, "geomagnetic_ohe_v2.pkl")

print("Upgraded model trained and saved successfully! 🚀")