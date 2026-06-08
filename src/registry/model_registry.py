import numpy as np
import pandas as pd
import dagshub
import mlflow
import boto3
import os
import tempfile
import joblib
import mlflow.sklearn
from src.cloud.s3_storage import S3Storage
from src.logger import setup_logger
from src.config.constant import BUCKET_NAME
from src.connections.mlflow_setup import setup_mlflow

logging = setup_logger()

class ModelRegistry:

    def __init__(self):
        self.bucket_name = BUCKET_NAME
        setup_mlflow()
        self._upload_to_s3 = S3Storage(bucket_name=self.bucket_name)._upload_to_s3
        
    def register(self, model, model_name: str, params: dict, metrics: dict, tags: dict = None) -> dict:
        tags = tags or {}

        with mlflow.start_run(tags={"model_name": model_name, **tags}) as run:
            run_id = run.info.run_id

            # # Log the hyperparameters
            # mlflow.log_params(params)
            # logging.info(f"Params logged: {params}")

            # Log metrics
            mlflow.log_metrics({
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"]
            })
            logging.info(
                f"Metrics logged: RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f} | R2: {metrics['r2']:.3f}"
            )

            # log and register the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
            logging.info(f"Model logged and registered as '{model_name}'")

            # Backup .joblib to the s3 bucket
            with tempfile.TemporaryDirectory() as tmpdir:
                local_path = os.path.join(tmpdir, f"{model_name}.joblib")
                joblib.dump(model, local_path) 
                s3_uri = self._upload_to_s3(local_path, model_name, run_id) 

        summary = {
            "run_id": run_id,
            "s3_uri": s3_uri,
            "model_name": model_name,
            "metrics": metrics
        }
        logging.info("Model registration completed.")
        return summary
