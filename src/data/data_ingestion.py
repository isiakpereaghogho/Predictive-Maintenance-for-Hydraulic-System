import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from src.config.constant import (
    sensor_data,
    maintenance_data,
    equipment_data,
    failure_data
)

from src.logger import setup_logger
from src.exceptions import CustomException

logger = setup_logger()

class DataIngestion:
    def __init__(self):
        self.sensor_path = sensor_data
        self.maintenance_path = maintenance_data
        self.equipment_path = equipment_data
        self.failure_path = failure_data

    def load_sensor_data(self):
        return pd.read_csv(self.sensor_path)

    def load_maintenance_data(self):
        return pd.read_csv(self.maintenance_path)

    def load_equipment_data(self):
        return pd.read_csv(self.equipment_path)

    def load_failure_data(self):
        return pd.read_csv(self.failure_path)

    def load_all_data(self):
        try:
            sensor_df = self.load_sensor_data()
            maintenance_df = self.load_maintenance_data()
            equipment_df = self.load_equipment_data()
            failure_df = self.load_failure_data()
            print(sensor_df.head())
            print(maintenance_df.head())
            print(equipment_df.head())
            print(failure_df.head())
            logger.info("Data loaded successfully")

            return sensor_df, maintenance_df, equipment_df, failure_df
        
        except Exception as e:
            logger.error(f"Error occurred while loading data: {e}")
            raise CustomException(e)

