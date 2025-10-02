import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("src/models/gradient_boosting_model.pkl")

# Load dataset for VM IDs
df = pd.read_csv("data/raw/cloud_metrics.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

st.title("Cloud VM CPU Usage Prediction")
st.write("Predict CPU usage for your cloud VM based on system metrics.")

# Select VM
vm_id = st.selectbox("Select VM ID", df['vm_id'].unique())

# Get last metrics for selected VM
vm_data = df[df['vm_id'] == vm_id].iloc[-1].copy()

# Fill missing numerical values with 0
num_cols = ['memory_usage', 'network_traffic', 'power_consumption',
            'num_executed_instructions', 'execution_time', 'energy_efficiency']
vm_data[num_cols] = vm_data[num_cols].fillna(0)

# Input sliders for numerical features
st.sidebar.header("Adjust Metrics")
for col in num_cols:
    vm_data[col] = st.sidebar.slider(
        label=col.replace("_", " ").title(),
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(vm_data[col])
    )

# Dropdowns for categorical features
cat_cols = ['task_type', 'task_priority', 'task_status']
st.sidebar.header("Categorical Features")
for col in cat_cols:
    vm_data[col] = st.sidebar.selectbox(
        label=col.replace("_", " ").title(),
        options=df[col].dropna().unique(),
        index=list(df[col].dropna().unique()).index(vm_data[col])
    )

# Feature selection for prediction
features = ['memory_usage', 'network_traffic', 'power_consumption',
            'num_executed_instructions', 'execution_time', 'energy_efficiency', 'hour']
# Add hour from timestamp
vm_data['hour'] = vm_data['timestamp'].hour

X_demo = vm_data[features].values.reshape(1, -1)

# Predict
cpu_pred = model.predict(X_demo)[0]

# Display results
st.subheader(f"Predicted CPU Usage for VM {vm_id}")
st.write(f"{cpu_pred:.2f}%")

st.subheader("Current System Metrics")
st.json({col: vm_data[col] for col in num_cols + cat_cols})
