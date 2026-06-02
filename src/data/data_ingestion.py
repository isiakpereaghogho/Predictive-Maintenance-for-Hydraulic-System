import os
import pandas as pd

from src.config.constant import (
    sensor_data,
    maintenance_data,
    equipment_data,
    failure_data
)


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
        sensor_df = self.load_sensor_data()
        maintenance_df = self.load_maintenance_data()
        equipment_df = self.load_equipment_data()
        failure_df = self.load_failure_data()

        return sensor_df, maintenance_df, equipment_df, failure_df


if __name__ == "__main__":
    ingestion = DataIngestion()

    sensor_df, maintenance_df, equipment_df, failure_df = ingestion.load_all_data()

    print("Sensor data shape:", sensor_df.shape)
    print("Maintenance data shape:", maintenance_df.shape)
    print("Equipment data shape:", equipment_df.shape)
    print("Failure data shape:", failure_df.shape)

    print("Sensor data:")
    print(sensor_df.head())
    print("Maintenance data:")
    print(maintenance_df.head())
    print("Equipment data:")
    print(equipment_df.head())
    print("Failure data:")
    print(failure_df.head())

    print("Sensor data null values:", sensor_df.isnull().sum())
    print("Maintenance data null values:", maintenance_df.isnull().sum())
    print("Equipment data null values:", equipment_df.isnull().sum())
    print("Failure data null values:", failure_df.isnull().sum())

    print("Sensor data duplicate values:", sensor_df.duplicated().sum())
    print("Maintenance data duplicate values:", maintenance_df.duplicated().sum())
    print("Equipment data duplicate values:", equipment_df.duplicated().sum())
    print("Failure data duplicate values:", failure_df.duplicated().sum())

    print("Sensor data dtypes:", sensor_df.dtypes)
    print("Maintenance data dtypes:", maintenance_df.dtypes)
    print("Equipment data dtypes:", equipment_df.dtypes)
    print("Failure data dtypes:", failure_df.dtypes)

    print("Sensor data info:")
    sensor_df.info()
    print("Maintenance data info:")
    maintenance_df.info()
    print("Equipment data info:")
    equipment_df.info()
    print("Failure data info:")
    failure_df.info()

    print("Sensor data describe:")
    print(sensor_df.describe())
    print("Maintenance data describe:")
    print(maintenance_df.describe())
    print("Equipment data describe:")
    print(equipment_df.describe())
    print("Failure data describe:")
    print(failure_df.describe())
