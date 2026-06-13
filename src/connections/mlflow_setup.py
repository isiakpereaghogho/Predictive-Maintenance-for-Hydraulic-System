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

# def setup_mlflow():
#     dagshub.init(
#     repo_owner=DAGSHUB_OWNER,
#     repo_name=DAGSHUB_REPO,
#     mlflow=True
#     )
#     mlflow.set_experiment(EXPERIMENT)
    
experiment_name = EXPERIMENT
def setup_mlflow():
    dagshub_token = os.getenv('bosch_prediction_env_Dagshub_token')
    if not dagshub_token:
        raise EnvironmentError('bosch_prediction_env_Dagshub_token environment is not set')
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_token
    os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

    dagshub_url = 'https://dagshub.com'
    repo_owner=DAGSHUB_OWNER
    repo_name=DAGSHUB_REPO
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    mlflow.set_experiment(experiment_name)


def load_best_model_by_r2(registered_model_name: str = "gb_model"):
    """
    Load the best MLflow registered model version based on highest R2 score.
    Returns:
        model: loaded sklearn model
        model_info: dictionary containing selected model metadata
    """

    client = MlflowClient()

    logging.info(f"Searching MLflow registered model versions for: {registered_model_name}")

    model_versions = list(
        client.search_model_versions(
            filter_string=f"name='{registered_model_name}'"
        )
    )

    if not model_versions:
        raise ValueError(
            f"No registered model versions found for '{registered_model_name}'."
        )

    best_model_info = None

    for mv in model_versions:
        try:
            run = client.get_run(mv.run_id)
            metrics = run.data.metrics

            r2 = (
                metrics.get("r2")
                or metrics.get("R2")
                or metrics.get("R²")
                or metrics.get("r_squared")
            )

            if r2 is None:
                logging.warning(
                    f"Skipping {registered_model_name} version {mv.version}: no R2 metric found."
                )
                continue

            r2 = float(r2)

            logging.info(
                f"Model={mv.name}, Version={mv.version}, Run={mv.run_id}, R2={r2}"
            )

            if best_model_info is None or r2 > best_model_info["r2"]:
                best_model_info = {
                    "model_name": mv.name,
                    "best_version": str(mv.version),
                    "best_run_id": mv.run_id,
                    "r2": r2,
                    "rmse": metrics.get("rmse") or metrics.get("RMSE"),
                    "mae": metrics.get("mae") or metrics.get("MAE"),
                    "source": "mlflow"
                }

        except Exception as e:
            logging.warning(
                f"Could not inspect version {mv.version} of {registered_model_name}: {e}"
            )

    if best_model_info is None:
        raise ValueError(
            f"No model version for '{registered_model_name}' has a valid R2 metric."
        )

    model_uri = (
        f"models:/{registered_model_name}/{best_model_info['best_version']}"
    )

    logging.info(f"Loading best model from MLflow URI: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)

    logging.info(
        f"Best model loaded successfully: "
        f"{registered_model_name} version {best_model_info['best_version']} "
        f"with R2={best_model_info['r2']}"
    )

    return model, best_model_info
    
# def load_best_model_by_r2(registered_model_name="gb_model"):
#     """
#     Load the best registered MLflow model version based on highest R2.
#     """

#     client = MlflowClient()

#     model_versions = client.search_model_versions(f"name='{registered_model_name}'")

#     if len(model_versions) == 0:
#             raise ValueError(
#                 f"No model versions found for {registered_model_name}"
#             )

#     best_r2 = float("-inf")
#     best_version = None
#     best_run_id = None

#     for mv in model_versions:
#         run = client.get_run(mv.run_id)
#         metrics = run.data.metrics

#         r2 = metrics.get("R2") or metrics.get("r2") or metrics.get("r_squared")

#     if r2 is not None and r2 > best_r2:
#                 best_r2 = r2
#                 best_version = mv.version
#                 best_run_id = mv.run_id

#     if best_version is None:
#             raise ValueError(
#                 f"No model version for {registered_model_name} has an R2 metric."
#             )

#     model_uri = f"models:/{registered_model_name}/{best_version}"

#     model = mlflow.sklearn.load_model(model_uri)

#     logging.info(
#             f"Best model loaded: {registered_model_name} version {best_version}"
#         )
#     logging.info(f"Best R2: {best_r2}")
#     logging.info(f"Best Run ID: {best_run_id}")

#     return model, {
#             "model_name": registered_model_name,
#             "best_version": best_version,
#             "best_r2": best_r2,
#             "best_run_id": best_run_id}