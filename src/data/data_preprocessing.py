import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from src.logger import setup_logger
from src.exceptions import CustomException
from src.data.data_ingestion import DataIngestion
import os

logging = setup_logger()

class DataPreprocessing:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.sensor_data, self.maintenance_data, self.equipment_data, self.failure_data = self.data_ingestion.load_all_data()
    
    def data_merging(self):
        try:
            self.sensor_data['timestamp'] = pd.to_datetime(self.sensor_data['timestamp'], format='%d/%m/%Y %H:%M',errors='coerce')
            self.maintenance_data['action_timestamp'] = pd.to_datetime(self.maintenance_data['action_timestamp'], format='%d/%m/%Y %H:%M', errors='coerce')
            self.equipment_data['installation_date'] = pd.to_datetime(self.equipment_data['installation_date'], format='%d/%m/%Y', errors='coerce')
            self.equipment_data['last_filter_change_date'] = pd.to_datetime(self.equipment_data['last_filter_change_date'], format='%d/%m/%Y', errors='coerce')
            self.failure_data['failure_timestamp'] = pd.to_datetime(self.failure_data['failure_timestamp'],dayfirst=True,errors='coerce')
            self.failure_data['degradation_start_timestamp'] = pd.to_datetime(self.failure_data['degradation_start_timestamp'],dayfirst=True,errors='coerce')

            logging.info("Timestamps converted to datetime successfully")
            logging.info(f"sensor_data: {len(self.sensor_data)} rows")
            logging.info(f"maintenance_data: {len(self.maintenance_data)} rows")
            logging.info(f"equipment_data: {len(self.equipment_data)} rows")
            logging.info(f"failure_data: {len(self.failure_data)} rows")

            # Merge equipment dataset to sensor dataset
            merged_sensor_data = self.sensor_data.merge(
                self.equipment_data,
                on='machine_id',
                how='left'
            )

            # Sort by timestamp first for merge_asof
            merged_sensor_data = merged_sensor_data.sort_values(
                ['timestamp', 'machine_id']
            ).reset_index(drop=True)

            self.maintenance_data = self.maintenance_data.sort_values(
                ['action_timestamp', 'machine_id']
            ).reset_index(drop=True)

            # Merge maintenance data
            merged_df = pd.merge_asof(
                merged_sensor_data,
                self.maintenance_data,
                left_on='timestamp',
                right_on='action_timestamp',
                by='machine_id',
                direction='backward'
            )

            logging.info(f"Missing failure timestamps before merge: " f"{self.failure_data['failure_timestamp'].isna().sum()}")

            self.failure_data = self.failure_data.dropna(subset=['failure_timestamp'])

            logging.info(f"failure_data after dropping invalid failure timestamps: " f"{self.failure_data.shape}")

            # Sort again before failure merge
            merged_df = merged_df.sort_values(
                ['timestamp', 'machine_id']
            ).reset_index(drop=True)

            self.failure_data = self.failure_data.sort_values(
                ['failure_timestamp', 'machine_id']
            ).reset_index(drop=True)

            # Merge failure data
            merged_data = pd.merge_asof(
                merged_df,
                self.failure_data,
                left_on='timestamp',
                right_on='failure_timestamp',
                by='machine_id',
                direction='backward'
            )
            logging.info("Data merged successfully")
            logging.info(f"merged_data: {len(merged_data)} rows")
            logging.info(merged_data.head())
            logging.info(f"Merged data shape: {merged_data.shape}")
            logging.info(f"Merged data columns: {merged_data.columns.tolist()}")
            logging.info(f"Missing values in merged_data:\n{merged_data.isnull().sum()}")

            return merged_data
        except Exception as e:
            logging.error(f"Error occurred during data merging: {e}")
            raise CustomException(e, sys)