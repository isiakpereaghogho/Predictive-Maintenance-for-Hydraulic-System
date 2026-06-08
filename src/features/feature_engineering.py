import sys
import io
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.config.constant import BUCKET_NAME, featured_data, featured_data_json
from src.logger import setup_logger
from src.exceptions import CustomException
from src.data.data_cleaning import DataCleaning
from src.cloud.s3_storage import S3Storage

logging = setup_logger()


class FeatureEngineering:
    def __init__(self):
        self.cleaner = DataCleaning()
        self.cleaned_data = self.cleaner.clean_data()

    def create_features(self):
        try:
            df = self.cleaned_data.copy()

            logging.info(f"Starting feature engineering. Shape: {df.shape}")

            # Ensure datetime columns
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['installation_date'] = pd.to_datetime(df['installation_date'], errors='coerce')
            df['last_filter_change_date'] = pd.to_datetime(df['last_filter_change_date'], errors='coerce')

            # Machine age
            df['machine_age_days'] = (
                df['timestamp'] - df['installation_date']
            ).dt.days

            # Days since filter change
            df['days_since_filter_change'] = (
                df['timestamp'] - df['last_filter_change_date']
            ).dt.days

            # Sensor columns
            sensor_cols = [
                'pressure_bar',
                'temp_celsius',
                'flow_lpm',
                'vibration_x_g',
                'vibration_y_g',
                'pump_rpm'
            ]

            # Vibration magnitude
            df['vibration_magnitude'] = np.sqrt(
                df['vibration_x_g'] ** 2 + df['vibration_y_g'] ** 2
            )

            # Equipment/sensor interaction features
            df['hydraulic_power'] = df['pressure_bar'] * df['flow_lpm']

            df['flow_efficiency'] = (
                df['flow_lpm'] / df['pump_rpm'].replace(0, np.nan)
            ).fillna(0)

            df['pressure_per_rpm'] = (
                df['pressure_bar'] / df['pump_rpm'].replace(0, np.nan)
            ).fillna(0)

            df['thermal_stress_index'] = (
                df['temp_celsius'] * df['vibration_magnitude']
            )

            df['vibration_flow_ratio'] = (
                df['vibration_magnitude'] / df['flow_lpm'].replace(0, np.nan)
            ).fillna(0)

            df['pump_efficiency_index'] = (
                (df['flow_lpm'] * df['pressure_bar']) /
                df['pump_rpm'].replace(0, np.nan)
            ).fillna(0)

            df['pressure_temp_ratio'] = (
                df['pressure_bar'] / df['temp_celsius'].replace(0, np.nan)
            ).fillna(0)

            # Delta features
            for col in sensor_cols:
                df[f'{col}_delta'] = (
                    df.groupby('machine_id')[col]
                    .diff()
                    .fillna(0)
                )

            delta_cols = [f'{col}_delta' for col in sensor_cols]

            # Lag features
            lag_steps = [1, 3, 6]

            for col in sensor_cols:
                for lag in lag_steps:
                    df[f'{col}_lag{lag}'] = (
                        df.groupby('machine_id')[col]
                        .shift(lag)
                        .bfill()
                    )

            lag_cols = [
                f'{col}_lag{lag}'
                for col in sensor_cols
                for lag in lag_steps
            ]

            # Rolling features
            window = 15

            for col in sensor_cols:
                grouped = df.groupby('machine_id')[col]

                df[f'{col}_roll_mean_15m'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

                df[f'{col}_roll_std_15m'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                ).fillna(0)

                df[f'{col}_roll_min_15m'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).min()
                )

                df[f'{col}_roll_max_15m'] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).max()
                )

            rolling_cols = [
                f'{col}_{stat}_15m'
                for col in sensor_cols
                for stat in ['roll_mean', 'roll_std', 'roll_min', 'roll_max']
            ]

            # Encode categorical columns
            if 'shift' in df.columns:
                le_shift = LabelEncoder()
                df['shift'] = le_shift.fit_transform(df['shift'].astype(str))

            if 'fluid_type' in df.columns:
                le_fluid = LabelEncoder()
                df['fluid_type'] = le_fluid.fit_transform(df['fluid_type'].astype(str))

            # Base features
            base_features = sensor_cols + [
                'machine_age_days',
                'day_of_week',
                'days_since_filter_change',
                'thermal_stress_index',
                'vibration_flow_ratio',
                'pump_efficiency_index',
                'pressure_temp_ratio',
                'pressure_per_rpm',
                'vibration_magnitude',
                'hydraulic_power',
                'flow_efficiency',
                'shift',
                'fluid_type'
            ]

            final_features_rul = base_features + delta_cols + lag_cols + rolling_cols

            # Drop redundant/leakage/unwanted columns
            cols_to_drop = [
                'thermal_stress_index',
                'vibration_flow_ratio',
                'pump_efficiency_index',
                'pump_rpm_lag1',
                'pump_rpm_lag3',
                'pump_rpm_lag6',
                'pump_rpm_roll_mean_15m',
                'flow_lpm_lag1',
                'flow_lpm_lag3',
                'flow_lpm_lag6',
                'flow_lpm_roll_mean_15m',
                'vibration_x_g_roll_mean_15m',
                'vibration_y_g_roll_mean_15m',
                'flow_lpm',
                'pressure_bar',
                'vibration_x_g',
                'vibration_y_g',
                'is_anomaly',
                'installation_date',
                'last_filter_change_date',
                'failure_mode_x',
                'total_operating_hours',
                'maintenance_priority',
                'days_since_filter_change',
                'is_sensor_dropout',
                'machine_age_days'
                'pressure_bar_roll_std_15m', 
                'pressure_bar_roll_min_15m', 
                'pressure_bar_roll_max_15m', 
                'temp_celsius_roll_std_15m', 
                'temp_celsius_roll_min_15m', 
                'temp_celsius_roll_max_15m', 
                'flow_lpm_roll_std_15m', 
                'flow_lpm_roll_min_15m', 
                'flow_lpm_roll_max_15m', 
                'vibration_x_g_roll_std_15m', 
                'vibration_x_g_roll_min_15m', 
                'vibration_x_g_roll_max_15m', 
                'vibration_y_g_roll_std_15m', 
                'vibration_y_g_roll_min_15m', 
                'vibration_y_g_roll_max_15m', 
                'pump_rpm_roll_std_15m', ''
                'pump_rpm_roll_min_15m', 
                'pump_rpm_roll_max_15m',
                'pressure_bar_roll_std_15m'
            ]

            final_features_rul = [col for col in final_features_rul if col not in cols_to_drop]

            rul_dataset = df[
                final_features_rul + ['machine_id', 'timestamp', 'rul_hours']
            ].copy()

            logging.info("Feature engineering completed successfully")
            logging.info(f"final_features_RUL: {len(final_features_rul)}")
            logging.info(f"RUL dataset shape: {rul_dataset.shape}")
            logging.info(f"RUL dataset columns: {rul_dataset.columns.tolist()}")

                      
            # storing of featured data in S3
            bucket_name = BUCKET_NAME
            s3 = S3Storage(bucket_name)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            csv_buffer = io.StringIO()
            rul_dataset.to_csv(csv_buffer, index=False)

            s3.upload_bytes(data=csv_buffer.getvalue(), s3_key=f"features/rul_dataset_{timestamp}.csv", content_type='text/csv')

            json_buffer = json.dumps({
                'FINAL_FEATURES_RUL': final_features_rul,
                'SENSOR_COLS': sensor_cols}, indent=2)
            s3.upload_bytes(data=json_buffer, s3_key=f"features/features_metadata{timestamp}.json", content_type='application/json')

            # Create folder if it does not exist
            os.makedirs(os.path.dirname(featured_data), exist_ok=True)

            # Save featured dataset
            rul_dataset.to_csv(featured_data, index=False)

            # Save feature columns
            with open(featured_data_json, "w") as f:
                json.dump(final_features_rul, f, indent=4)

            logging.info(f"Featured dataset saved to: {featured_data}")
            logging.info(f"Feature columns saved to: {featured_data_json}")

            return rul_dataset, final_features_rul
        
        except Exception as e:
            logging.error(f"Error occurred during feature engineering: {e}")
            raise CustomException(e, sys)   

if __name__ == "__main__":
    features = FeatureEngineering()
    rul_dataset, final_features_rul = features.create_features()

    print(rul_dataset.head())
    print(rul_dataset.shape)
    print(len(final_features_rul))
    print(final_features_rul)
    
    