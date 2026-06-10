from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
from src.connections.mlflow_setup import setup_mlflow, load_best_model_by_r2
from src.cloud.s3_storage import S3Storage
from src.config.constant import BUCKET_NAME
from src.pipeline.prediction import predict_rul
from src.pipeline.training import TrainingPipeline
from src.logger import setup_logger
from src.exceptions import CustomException
from mlflow.tracking import MlflowClient



logging = setup_logger()

app = FastAPI(title="RUL Prediction API")

setup_mlflow()

s3 = S3Storage(BUCKET_NAME)

model_info = {}


def load_artifacts():
    try:

        client = MlflowClient()

        versions = client.search_model_versions("name='gb_model'")

        latest_version = max(int(v.version) for v in versions)

        model, model_info = load_best_model_by_r2("gb_model")
        
        logging.info(f"Loaded gb_model version {latest_version}")

        json_key = s3.get_latest_file(prefix="features/",keyword="features_metadata")

        feature_cols = s3.load_json(json_key)["FINAL_FEATURES_RUL"]

        csv_key = s3.get_latest_file(prefix="features/",keyword="rul_dataset")

        dataset = s3.load_csv(csv_key)
        dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])

        return model, dataset, feature_cols,  model_info

    except Exception as e:
        logging.error(f"Error loading artifacts: {e}")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to load artifacts: {e}"
        )


# load at startup
@app.on_event("startup")
def startup_event():
    global model, dataset, feature_cols

    model, dataset, feature_cols,  model_info = load_artifacts()

    logging.info("Artifacts loaded successfully")

# REQUEST SCHEMA
class SensorInput(BaseModel):
    machine_id: str
    pressure_bar: float
    temp_celsius: float
    flow_lpm: float
    vibration_x_g: float
    vibration_y_g: float
    pump_rpm: float


@app.get("/")
def home():
    return {"message": "RUL Prediction API is running"}

@app.get("/model-info")
def get_model_info():
    return {
        "model_name": model_info.get("MODEL_NAME"),
        "version": model_info.get("best_version"),
        "r2": model_info.get("best_r2"),
        "run_id": model_info.get("best_run_id")
    }

# training route
@app.post("/train")
def train_model():
    global model, dataset, feature_cols

    try:
        pipeline = TrainingPipeline()
        pipeline.train()

        # Reloaded updated artifacts
        model, dataset, feature_cols = load_artifacts()

        return {
            "status": "Model training completed and artifacts"
        }
    
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
    
# prediction route
@app.post("/predict")
def predict(data: SensorInput):

    sensor_input = {
        "pressure_bar": data.pressure_bar,
        "temp_celsius": data.temp_celsius,
        "flow_lpm": data.flow_lpm,
        "vibration_x_g": data.vibration_x_g,
        "vibration_y_g": data.vibration_y_g,
        "pump_rpm": data.pump_rpm,
    }
    
    try:
        result = predict_rul(
            machine_id=data.machine_id,
            sensor_input=sensor_input,
            model=model,
            dataset=dataset,
            feature_cols=feature_cols,
        )
    
        return result

    except ValueError as e:
        # Known issues
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logging.error(f"Prediction failed: {e}")

    raise HTTPException(
        status_code=500,
        detail=f"Prediction failed: {e}"
    )