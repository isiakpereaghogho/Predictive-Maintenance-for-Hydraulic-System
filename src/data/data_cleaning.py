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
from src.config.constant import cleaned_data as DATA_DIR
from src.data.data_preprocessing import DataPreprocessing

logging = setup_logger()

class DataCleaning:
    def __init__(self):
        self.processor = DataPreprocessing()
        self.merged_data = self.processor.data_merging()

    def clean_data(self):
        try:
            # Use the merged data from preprocessing
            merged_df = self.merged_data.copy()
            logging.info(f"Initial merged data shape: {merged_df.shape}")
            logging.info(f"Missing values in merged data:\n{merged_df.isnull().sum()}")
            merged_df = merged_df.sort_values(by =['machine_id', 'timestamp']).reset_index(drop=True)

            # Interpolate missing sensor values
            sensor_cols = ['pressure_bar', 'temp_celsius', 'flow_lpm', 'vibration_x_g', 'vibration_y_g', 'pump_rpm']
            merged_df[sensor_cols] = (merged_df.groupby('machine_id')[sensor_cols].transform(lambda x: x.interpolate(method='linear',limit_direction='both')))

            logging.info(f"Missing values after interpolation:\n{merged_df.isnull().sum()}")

            #Downsizing the dataset by keeping only rows whereanomalies equal to 1
            merged_df = merged_df[merged_df['is_anomaly'] == 1].copy()
            #Dropping columns with 100% missing values
            drop_cols = ['failure_event_id','failure_timestamp','failure_mode_y','degradation_start_timestamp','repair_cost_usd','downtime_hours']

            merged_df = merged_df.drop(columns=drop_cols)

            logging.info("Dropped 100% missing columns")
            logging.info(f"Columns dropped: {drop_cols}")
            logging.info("Keeping only rows for RUL modelling")
            logging.info(f"Data shape after filtering anomalies: {merged_df.shape}")

            # Save cleaned data to CSV
            os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
            merged_df.to_csv(DATA_DIR, index=False)
            logging.info(f"Cleaned data saved to {DATA_DIR}")

            return merged_df
        
        except Exception as e:
            logging.error(f"Error occurred during data cleaning: {e}")
            raise CustomException(e, sys)