## Cloud CPU Usage Forecasting
Predict CPU usage for cloud VMs based on system metrics using machine learning regression models.

## Project Overview
This project aims to forecast the CPU utilization (%) of virtual machines in a cloud environment using historical system metrics. Accurate predictions help optimize resource allocation, improve performance, and reduce energy consumption.

The dataset contains 100,000 rows of VM metrics, including memory usage, network traffic, power consumption, execution time, task type, and priority, collected over a period of time.

## Features Used
- Numerical Metrics: memory_usage, network_traffic, power_consumption, num_executed_instructions, execution_time, energy_efficiency
- Categorical Metrics: task_type, task_priority, task_status
- Time Features: hour, day_of_week, month, is_weekend

## Lag & Rolling Statistics: cpu_usage_lag_1, cpu_usage_lag_5, cpu_usage_roll_mean_3, cpu_usage_roll_std_3, etc.

## Machine Learning Models
- Baseline: Linear Regression / Mean Prediction

## Advanced Models:
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Evaluation Metrics: RMSE, MAE, R²

## Key Findings
- CPU usage and other VM metrics exhibit near-uniform distribution, suggesting synthetic dataset characteristics.
- Lag and rolling statistics improve predictive power significantly.