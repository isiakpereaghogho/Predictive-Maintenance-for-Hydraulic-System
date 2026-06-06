import os
import pandas as pd

BASE_DIR = BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
equipment_data = os.path.join(BASE_DIR, 'Dataset', 'equipment_master.csv')
failure_data = os.path.join(BASE_DIR, 'Dataset', 'failure_labels.csv')
sensor_data = os.path.join(BASE_DIR, 'Dataset', 'sensor_telemetry_cleaneed.csv')
maintenance_data = os.path.join(BASE_DIR, 'Dataset', 'maintenance_log.csv')

cleaned_data = os.path.join(BASE_DIR, 'Dataset', 'cleaned_data', 'cleaned_data.csv')
featured_data = os.path.join(BASE_DIR, 'Dataset', 'featured_data', 'rul_featured_data.csv')
featured_data_json = os.path.join(BASE_DIR, 'Dataset', 'featured_data', 'bosch_featured_set.json')

BUCKET_NAME = 'bosch-predictive-maintenance'
MODEL_NAME = 'bosch_predictive_maintenance_model'
S3_ARTIFACT_ROOT = f's3://{BUCKET_NAME}/mlflow_artifacts'
DAGSHUB_OWNER = 'isiakpereaghogho'
DAGSHUB_REPO = 'Predictive-Maintenance-for-Hydraulic-System'
EXPERIMENT = 'bosch_predictive_maintenance'