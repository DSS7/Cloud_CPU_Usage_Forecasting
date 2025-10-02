import pandas as pd
import joblib
import numpy as np

# Load new data
new_data = pd.read_csv(r"C:\Users\37789\OneDrive\Documents\GitHub\Cloud_CPU_Usage_Forecasting\data\raw\new_cloud_metrics.csv")

# Feature engineering for new data (lag/rolling/hour/day/month/etc.)
new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
new_data['hour'] = new_data['timestamp'].dt.hour
new_data['day_of_week'] = new_data['timestamp'].dt.dayofweek
new_data['month'] = new_data['timestamp'].dt.month
new_data['is_weekend'] = new_data['day_of_week'].isin([5,6]).astype(int)

# Load the trained model
model_path = r"C:\Users\37789\OneDrive\Documents\GitHub\Cloud_CPU_Usage_Forecasting\src\models\gradient_boosting_model.pkl"
model = joblib.load(model_path)

# Select features used in training
features = [
    'cpu_usage_lag_1', 'cpu_usage_roll_mean_3', 'cpu_usage_roll_max_3',
    'num_executed_instructions', 'network_traffic', 'cpu_usage_lag_5', 'hour'
]

X_new = new_data[features]

# Predict CPU usage
predictions = model.predict(X_new)

# Attach predictions to dataframe
new_data['predicted_cpu_usage'] = predictions

# Show first few predictions
print(new_data[['timestamp', 'cpu_usage', 'predicted_cpu_usage']].head())
