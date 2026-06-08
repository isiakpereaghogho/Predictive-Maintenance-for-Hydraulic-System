import json
import pandas as pd
import mlflow.sklearn
import dagshub
from src.pipeline.prediction import predict_rul
from src.config.constant import featured_data, featured_data_json,DAGSHUB_OWNER, DAGSHUB_REPO, EXPERIMENT

from src.pipeline.prediction import predict_rul


# Connect to DagsHub MLflow first
dagshub.init(
    repo_owner=DAGSHUB_OWNER,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)

mlflow.set_experiment(EXPERIMENT)


# Load model
model = mlflow.sklearn.load_model("models:/gb_model/5")

# Load featured dataset
dataset = pd.read_csv(featured_data)
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])

# Load feature columns
with open(featured_data_json, "r") as f:
    feature_cols = json.load(f)

# Example sensor input
sensor_input = {
    "pressure_bar": 130.5,
    "temp_celsius": 58.2,
    "flow_lpm": 42.7,
    "vibration_x_g": 0.18,
    "vibration_y_g": 0.22,
    "pump_rpm": 1450
}

# Pick an existing machine_id from your dataset
machine_id = dataset['machine_id'].iloc[0]

# Test prediction
result = predict_rul(
    machine_id=machine_id,
    sensor_input=sensor_input,
    model=model,
    dataset=dataset,
    feature_cols=feature_cols
)

print(result)