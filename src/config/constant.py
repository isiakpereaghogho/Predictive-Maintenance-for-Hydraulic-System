import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
CLEANED_DATA_DIR = os.path.join(DATASET_DIR, "cleaned_data")
FEATURED_DATA_DIR = os.path.join(DATASET_DIR, "featured_data")

os.makedirs(CLEANED_DATA_DIR, exist_ok=True)
os.makedirs(FEATURED_DATA_DIR, exist_ok=True)

equipment_data = os.path.join(DATASET_DIR, "equipment_master.csv")
failure_data = os.path.join(DATASET_DIR, "failure_labels.csv")
sensor_data = os.path.join(DATASET_DIR, "sensor_telemetry_cleaneed.csv")
maintenance_data = os.path.join(DATASET_DIR, "maintenance_log.csv")

cleaned_data = os.path.join(CLEANED_DATA_DIR, "cleaned_data.csv")
featured_data = os.path.join(FEATURED_DATA_DIR, "rul_featured_data.csv")
featured_data_json = os.path.join(FEATURED_DATA_DIR, "featured_set.json")

BUCKET_NAME = "group-one-featured-engineer"
MODEL_NAME = "gb_model"
MODEL_STAGE = "5"

S3_ARTIFACT_ROOT = f"s3://{BUCKET_NAME}/mlflow_artifacts"

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
LOCAL_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")

DAGSHUB_OWNER = "isiakpereaghogho"
DAGSHUB_REPO = "Predictive-Maintenance-for-Hydraulic-System"
EXPERIMENT = "bosch_predictive_maintenance"

