import sys
import os
import dagshub
import mlflow
import mlflow.sklearn
import joblib
import boto3
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient
from src.config.constant import DAGSHUB_OWNER, DAGSHUB_REPO, EXPERIMENT
from src.logger import setup_logger
from src.exceptions import CustomException

from dotenv import load_dotenv
load_dotenv(override=True)


logging = setup_logger()

def setup_mlflow():
    dagshub.init(
    repo_owner=DAGSHUB_OWNER,
    repo_name=DAGSHUB_REPO,
    mlflow=True
    )
    mlflow.set_experiment(EXPERIMENT)
    
# experiment_name = EXPERIMENT
# def setup_mlflow():
#     dagshub_token = os.getenv('bosch_prediction_env_Dagshub_token')
#     if not dagshub_token:
#         raise EnvironmentError('bosch_prediction_env_Dagshub_token environment is not set')
    
#     os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
#     os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

#     dagshub_url = 'https://dagshub.com'
#     repo_owner=DAGSHUB_OWNER
#     repo_name=DAGSHUB_REPO
#     # Set up MLflow tracking URI
#     mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
#     mlflow.set_experiment(experiment_name)
    