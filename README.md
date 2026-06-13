# Predictive Maintenance for Hydraulic System
The Predictive Maintenance project for Bosch Rexroth AG focuses on developing an intelligent machine learning system that predicts hydraulic equipment failures before breakdowns occur.

# Predictive Maintenance for Hydraulic Systems

## Project Overview

This project develops an end-to-end machine learning and MLOps solution for predicting the **Remaining Useful Life (RUL)** of hydraulic systems used by **Bosch Rexroth AG**. The objective is to enable proactive maintenance by identifying equipment degradation before failure occurs, reducing unplanned downtime, maintenance costs, and operational risks.

The solution combines data engineering, machine learning, cloud deployment, MLOps practices, and interactive visualization to deliver real-time maintenance intelligence.

## Business Problem

Hydraulic systems operate under demanding industrial conditions and are subject to wear, component degradation, contamination, and performance loss over time.

Traditional maintenance strategies are often:

* Reactive (fix after failure)
* Time-based (scheduled maintenance regardless of condition)

These approaches can result in:

* Unexpected equipment failures
* Increased maintenance costs
* Reduced equipment lifespan
* Production downtime

This project introduces a predictive maintenance framework capable of estimating the Remaining Useful Life (RUL) of hydraulic systems using machine learning and sensor telemetry data.

## Objectives

* Predict Remaining Useful Life (RUL) of hydraulic systems.
* Detect degradation patterns before failure occurs.
* Support proactive maintenance scheduling.
* Reduce operational downtime and maintenance costs.
* Provide real-time monitoring and prediction capabilities.
* Deploy a production-ready MLOps solution on AWS.

## Dataset

The project utilizes multiple data sources:

### Sensor Telemetry

* Pressure (bar)
* Temperature (°C)
* Flow Rate (LPM)
* Vibration X
* Vibration Y
* Pump RPM

### Equipment Information

* Installation Date
* Operating Hours
* Fluid Type
* Maintenance Priority

### Maintenance Logs

* Maintenance Events
* Repair Activities
* Filter Replacement Records

### Failure Records

* Failure Mode
* Failure Timestamp
* Downtime Duration
* Repair Cost

## Project Architecture

### Data Pipeline

1. Data Ingestion
2. Data Cleaning
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Model Registry (MLflow + DagsHub)
7. Docker Containerization
8. CI/CD Automation
9. AWS Deployment

### Cloud Infrastructure

* **Amazon S3** – Data storage and model artifacts
* **Amazon ECR** – Docker image registry
* **Amazon EC2** – Hosting FastAPI and Streamlit applications
* **GitHub Actions** – CI/CD pipeline
* **MLflow + DagsHub** – Experiment tracking and model registry

## Feature Engineering

The following feature groups were created:

### Statistical Features

* Rolling Mean
* Rolling Standard Deviation
* Rolling Minimum
* Rolling Maximum

### Time-Series Features

* Lag Features
* Delta Features

### Domain Features

* Hydraulic Power
* Flow Efficiency
* Pressure per RPM
* Pressure-Temperature Ratio
* Vibration Magnitude


## Machine Learning Models Evaluated

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* Gradient Boosting Regressor

### Best Performing Model

**Gradient Boosting Regressor**

Performance Metrics:

| Metric   | Value       |
| -------- | ----------- |
| R² Score | 0.933       |
| RMSE     | 14.76 Hours |
| MAE      | 11.42 Hours |


## MLOps Workflow

### Experiment Tracking

MLflow and DagsHub were used for:

* Experiment tracking
* Metric logging
* Parameter logging
* Model versioning
* Model comparison

### CI/CD Pipeline

GitHub Actions automatically:

1. Detects code changes.
2. Runs validation and testing.
3. Builds Docker images.
4. Pushes images to Amazon ECR.
5. Deploys updated containers to Amazon EC2.

## API Development

A FastAPI service was developed to expose prediction endpoints.

### Endpoints

#### Health Check

```http
GET /
```

#### Train Model

```http
POST /train
```

#### Predict Remaining Useful Life

```http
POST /predict
```

Example Request:

```json
{
  "machine_id": "HPU_01",
  "pressure_bar": 185,
  "temp_celsius": 62,
  "flow_lpm": 78,
  "vibration_x_g": 4.2,
  "vibration_y_g": 3.8,
  "pump_rpm": 1500
}

## Dashboard

A Streamlit dashboard was developed to provide:

* Real-time RUL predictions
* Equipment health monitoring
* KPI visualization
* Maintenance recommendations
* Operational insights

## Business Value

The solution delivers measurable business benefits:

* Reduced unplanned downtime
* Lower maintenance costs
* Increased equipment reliability
* Extended asset lifespan
* Improved maintenance planning
* Data-driven operational decisions

## Challenges

* Missing sensor readings and data quality issues.
* Limited failure events for supervised learning.
* Complex feature engineering requirements.
* Cloud deployment and IAM permission configuration.
* Model versioning and environment compatibility challenges.

## Future Improvements

* Real-time sensor streaming using AWS IoT Core.
* Advanced models such as XGBoost, LightGBM, and LSTM.
* Automated retraining pipelines.
* Model drift monitoring.
* Enterprise-scale deployment across multiple facilities.

## Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-Learn

### MLOps

* MLflow
* DagsHub
* Docker
* GitHub Actions

### Cloud

* AWS S3
* AWS ECR
* AWS EC2

### Deployment

* FastAPI
* Streamlit

**Building data-driven maintenance systems for smarter, more reliable industrial operations.**
