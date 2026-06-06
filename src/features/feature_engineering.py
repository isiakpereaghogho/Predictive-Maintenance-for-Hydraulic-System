import pandas as pd
import numpy as np
from pyparsing import col
from sklearn.preprocessing import LabelEncoder
from src.logger import setup_logger
from src.exceptions import CustomException
import os
from src.config.constant import featured_data_path, featured_data_json
from src.data.data_cleaning import DataCleaning
from datetime import datetime
import io
import json
from src.cloud.s3_storage import S3Storage
from sklearn.preprocessing import LabelEncoder

logging = setup_logger()

class FeatureEngineering:
    def __init__(self):
        self.cleaner = DataCleaning()
        self.cleaned_data = self.cleaner.clean_data().clean_data()

    def create_features(self):
        try:
            logging.info(f"Starting feature engineering on cleaned data with shape: {self.cleaned_data.shape}")
            logging.info(f"Columns in cleaned data: {self.cleaned_data.columns.tolist()}")

            # Create RUL feature
            self.cleaned_data['installation_date'] = pd.to_datetime(self.cleaned_data['installation_date'])
            self.cleaned_data['timestamp'] = pd.to_datetime(self.cleaned_data['timestamp'])
            self.cleaned_data['last_filter_change_date'] = pd.to_datetime(self.cleaned_data['last_filter_change_timestamp'])

            # Machine age in days at time of reading
            self.cleaned_data['machine_age_days'] = (
                self.cleaned_data['timestamp'] - self.cleaned_data['installation_date']
            ).dt.days

            # Days since last filter change
            self.cleaned_data['days_since_filter_change'] = (
                self.cleaned_data['timestamp'] - self.cleaned_data['last_filter_change_date']
            ).dt.days

            # Vibration Magnitude for combined mechanical stress
            self.cleaned_data['vibration_magnitude'] = np.sqrt(
                self.cleaned_data['vibration_x_g']**2 + self.cleaned_data['vibration_y_g']**2
            )

            # Hydraulic Power i.e pressure × flow
            self.cleaned_data['hydraulic_power'] = (
                self.cleaned_data['pressure_bar'] * self.cleaned_data['flow_lpm']
            )

            # Flow Efficiency i.e flow delivered per RPM
            self.cleaned_data['flow_efficiency'] = (
                self.cleaned_data['flow_lpm'] / self.cleaned_data['pump_rpm'].replace(0, np.nan)
            ).fillna(0)

            # Pressure per RPM for pressure generated per revolution
            self.cleaned_data['pressure_per_rpm'] = (
                self.cleaned_data['pressure_bar'] / self.cleaned_data['pump_rpm'].replace(0, np.nan)
            ).fillna(0)

            # Thermal Stress Index i.e heat × vibration combined stress
            self.cleaned_data['thermal_stress_index'] = (
                self.cleaned_data['temp_celsius'] * self.cleaned_data['vibration_magnitude']
            )

            # Vibration to Flow Ratio i.e cavitation risk indicator
            self.cleaned_data['vibration_flow_ratio'] = (
                self.cleaned_data['vibration_magnitude'] / self.cleaned_data['flow_lpm'].replace(0, np.nan)
            ).fillna(0)

            # Pump Efficiency Index for overall pump health
            self.cleaned_data['pump_efficiency_index'] = (
                (self.cleaned_data['flow_lpm'] * self.cleaned_data['pressure_bar']) / self.cleaned_data['pump_rpm'].replace(0, np.nan)
            ).fillna(0)

            # Pressure Temperature Ratio i.e the fluid/cooling health
            self.cleaned_data['pressure_temp_ratio'] = (
                self.cleaned_data['pressure_bar'] / self.cleaned_data['temp_celsius'].replace(0, np.nan)
            ).fillna(0)

            self.cleaned_data = self.cleaned_data.drop(columns=['installation_date', 'last_filter_change_date', 'maintenance_priority'])

            sensor_cols = ['pressure_bar', 'temp_celsius', 'flow_lpm',
               'vibration_x_g', 'vibration_y_g', 'pump_rpm']
            
            # Create delta features for sensor readings to capture trends
            for col in sensor_cols:
                self.cleaned_data[f'{col}_delta'] = (
                    self.cleaned_data.groupby('machine_id')[col].diff().fillna(0))
            delta_cols = [f'{col}_delta' for col in sensor_cols]
            logging.info(f"Delta features created: {delta_cols}")
            logging.info(f"Total delta features : {len(delta_cols)}")

            # Engineering lag features
            lag_steps = [1, 3, 6]

            for col in sensor_cols:
                for lag in lag_steps:
                    self.cleaned_data[f'{col}_lag{lag}'] = (self.cleaned_data.groupby('machine_id')[col].shift(lag).fillna(method='bfill'))

            lag_cols = [f'{col}_lag{lag}' for col in sensor_cols for lag in lag_steps]

            logging.info("Lag features created")
            logging.info(f"Total lag features: {len(lag_cols)}")
            for col in lag_cols:
                logging.info(f"{col}")

            #Creating rolling window features to capture short term trends and patterns
            WINDOW = 15
            for col in sensor_cols:
                grouped = self.cleaned_data.groupby('machine_id')[col]

            self.cleaned_data[f'{col}_roll_mean_15m'] = grouped.transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())

            rolling_cols = [f'{col}_{stat}_15m'
                for col in sensor_cols
                for stat in ['roll_mean', 'roll_std', 'roll_min', 'roll_max']]

            logging.info("Rolling window features created (window=15 mins)\n")
            logging.info(f"Total rolling features: {len(rolling_cols)}\n")
            for col in rolling_cols:
                logging.info(f"{col}")

            base_features = sensor_cols + ['machine_age_days', 'days_since_filter_change', 'thermal_stress_index', 
            'vibration_flow_ratio', 'pump_efficiency_index', 'pressure_temp_ratio', 'pressure_per_rpm', 
            'vibration_magnitude', 'hydraulic_power', 'flow_efficiency'] 
            
            final_features_rul = base_features + delta_cols + lag_cols + rolling_cols
            logging.info(f"Total features engineered: {len(final_features_rul)}")

            rul_dataset = self.cleaned_data[final_features_rul + ['machine_id', 'timestamp', 'rul_hours', 'shift', 'fluid_type']]
            logging.info("RUL feature created successfully")

            # #Label encoding of categorical features to perform correlation analysis and enable model training
            # # shift
            # le_shift = LabelEncoder()
            # self.cleaned_data['shift'] = le_shift.fit_transform(self.cleaned_data['shift'])

            # # fluid_type
            # le_fluid = LabelEncoder()
            # self.cleaned_data['fluid_type'] = le_fluid.fit_transform(self.cleaned_data['fluid_type'])

            # storing of featured data in S3
            bucket_name = 'bosch-predictive-maintenance'
            s3 = S3Storage(bucket_name)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            csv_buffer = io.StringIO()
            rul_dataset.to_csv(csv_buffer, index=False)

            s3.upload_bytes(csv_buffer.getvalue().encode(), featured_data_path)

            # Save featured data to CSV
            os.makedirs(os.path.dirname(featured_data_path), exist_ok=True)
            self.cleaned_data.to_csv(featured_data_path, index=False)
            logging.info(f"Featured data saved to {featured_data_path}")

            # Save featured data to JSON and upload to S3
            featured_json_buffer = io.StringIO()
            self.cleaned_data.to_json(featured_json_buffer, orient='records', date_format='iso')
            self.s3_storage.upload_fileobj(io.BytesIO(featured_json_buffer.getvalue().encode()), featured_data_json)
            logging.info(f"Featured data uploaded to S3 at {featured_data_json}")

        except Exception as e:
            logging.error(f"Error occurred during feature engineering: {e}")
            raise CustomException(e)
cleaner = DataCleaning()