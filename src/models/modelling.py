import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow.sklearn
from src.cloud.s3_storage import S3Storage
from src.features.feature_engineering import FeatureEngineering
from src.logger import setup_logger
from src.registry.model_registry import ModelRegistry
from src.config.constant import BUCKET_NAME

logging = setup_logger()

class ModellingPipeline:
    def __init__(self, bucket_name=BUCKET_NAME):
        self.s3 = S3Storage(bucket_name)

        self.rul_data = None
        self.feature_set = None
        self.FINAL_FEATURES_RUL = None
        self.SENSOR_COLS =  None

        self.model = None

    def run_feature_engineering(self):
        logging.info("Starting feature engineering")
        FeatureEngineering().engineer_features()
        logging.info("Feature engineering complete. Data uploaded to S3")

    def load_data(self):
        csv_key = self.s3.get_latest_file(
            prefix="features/",
            keyword="rul_dataset"
        )

        json_key = self.s3.get_latest_file(
            prefix="features/",
            keyword="features_metadata"
        )

        self.rul_data = self.s3.load_csv(csv_key)
        self.feature_set = self.s3.load_json(json_key)

        self.FINAL_FEATURES_RUL = self.feature_set["FINAL_FEATURES_RUL"]
        self.SENSOR_COLS = self.feature_set["SENSOR_COLS"]

        logging.info(f"Loaded data: {len(self.rul_data):,} rows")

        def prepare_data(self):
            self.rul_data = self.rul_data.sort_values("timestamp")

            split_time = self.rul_data["timestamp"].quantile(0.8)

            train = self.rul_data[self.rul_data["timestamp"] <= split_time]
            test = self.rul_data[self.rul_data["timestamp"] > split_time]

            train = train.dropna(subset=self.FINAL_FEATURES_RUL + ["rul_hours"])
            test = test.dropna(subset=self.FINAL_FEATURES_RUL + ["rul_hours"])

            logging.info(f"Train: {len(self.X_train):,} rows | Test: {len(self.X_test):,} rows")

            def train_model(self):
                logging.info("Training model")
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    subsample=0.8,
                    random_state=42
                )
                self.mdoel.fit(self.X_train, self.y_train)
                logging.info("Model training complete")


            def evaluate(self):
                preds = self.model.predict(self.X_test)

                rmse = np.sqrt(mean_squared_error(self.y_test, preds))
                mae = mean_absolute_error(self.y_test, preds)
                r2 = r2_score(self.y_test, preds)

                logging.info("\nModel Performance:")
                logging.info(f"RMSE: {rmse: .2f}")
                logging.info(f"MAE:  {mae: 2f}")
                logging.info(f"R2:  {r2: 3f}")


                return rmse, mae, r2, self.model

            def run(self):
                self.run_feature_engineering()
                self.load_data()
                self.prepare_data()
                self.train_model()

                rmse, mae, r2 self.model = self.evaluate()

                # hand off to registry
                registry = ModelRegistry()

                registry.register(
                    model = self.model,
                    model_name = "GradientBoosting_RUL",
                    params = {
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        subsample=0.8,
                        random_state=42
                    },
                    metrics={"rmse": rmse, "mae": mae, "r2": r2},
                    tags={"stage": "dev", "dataset": "hydraulic_system"},
                )

            