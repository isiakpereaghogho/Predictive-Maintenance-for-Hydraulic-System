import pandas as pd
import numpy as np


def predict_rul(
    machine_id: str,
    sensor_input: dict,
    model,
    dataset: pd.DataFrame,
    feature_cols: list
) -> dict:

    sensor_cols = [
        'pressure_bar',
        'temp_celsius',
        'flow_lpm',
        'vibration_x_g',
        'vibration_y_g',
        'pump_rpm'
    ]

    machine_df = dataset[dataset['machine_id'] == machine_id].copy()

    if machine_df.empty:
        raise ValueError(f"Machine {machine_id} not found")

    machine_df['timestamp'] = pd.to_datetime(machine_df['timestamp'])
    machine_df = machine_df.sort_values("timestamp")

    latest = machine_df.iloc[-1]

    new_row = {
        "machine_id": machine_id,
        "timestamp": pd.Timestamp.now(),
        **sensor_input
    }

    new_row['vibration_magnitude'] = np.sqrt(
        sensor_input['vibration_x_g'] ** 2 +
        sensor_input['vibration_y_g'] ** 2
    )

    new_row['hydraulic_power'] = (
        sensor_input['pressure_bar'] *
        sensor_input['flow_lpm']
    )

    new_row['flow_efficiency'] = (
        sensor_input['flow_lpm'] /
        sensor_input['pump_rpm']
        if sensor_input['pump_rpm'] != 0 else 0
    )

    new_row['pressure_per_rpm'] = (
        sensor_input['pressure_bar'] /
        sensor_input['pump_rpm']
        if sensor_input['pump_rpm'] != 0 else 0
    )

    new_row['vibration_flow_ratio'] = (
        new_row['vibration_magnitude'] /
        sensor_input['flow_lpm']
        if sensor_input['flow_lpm'] != 0 else 0
    )

    new_row['pump_efficiency_index'] = (
        (sensor_input['flow_lpm'] * sensor_input['pressure_bar']) /
        sensor_input['pump_rpm']
        if sensor_input['pump_rpm'] != 0 else 0
    )

    new_row['pressure_temp_ratio'] = (
        sensor_input['pressure_bar'] /
        sensor_input['temp_celsius']
        if sensor_input['temp_celsius'] != 0 else 0
    )

    new_row['machine_age_days'] = latest.get('machine_age_days', 0)
    new_row['days_since_filter_change'] = latest.get(
        'days_since_filter_change',
        0
    )

    history = machine_df.tail(15)

    combined = pd.concat(
        [history, pd.DataFrame([new_row])],
        ignore_index=True
    )

    combined = combined.sort_values(['machine_id', 'timestamp'])

    for col in sensor_cols:

        combined[f'{col}_delta'] = (
            combined.groupby('machine_id')[col]
            .diff()
            .fillna(0)
        )

        combined[f'{col}_lag1'] = (
            combined.groupby('machine_id')[col]
            .shift(1)
        )

        combined[f'{col}_lag3'] = (
            combined.groupby('machine_id')[col]
            .shift(3)
        )

        combined[f'{col}_lag6'] = (
            combined.groupby('machine_id')[col]
            .shift(6)
        )

        combined[f'{col}_roll_mean_15m'] = (
            combined.groupby('machine_id')[col]
            .transform(lambda x: x.rolling(15, min_periods=1).mean())
        )

        combined[f'{col}_roll_std_15m'] = (
            combined.groupby('machine_id')[col]
            .transform(lambda x: x.rolling(15, min_periods=1).std())
            .fillna(0)
        )

        combined[f'{col}_roll_min_15m'] = (
            combined.groupby('machine_id')[col]
            .transform(lambda x: x.rolling(15, min_periods=1).min())
        )

        combined[f'{col}_roll_max_15m'] = (
            combined.groupby('machine_id')[col]
            .transform(lambda x: x.rolling(15, min_periods=1).max())
        )

    feature_row = combined.iloc[[-1]].copy()

    for col in feature_cols:
        if col not in feature_row.columns:
            feature_row[col] = 0

    feature_row = feature_row[feature_cols].fillna(0)

    rul_hours = max(0.0, model.predict(feature_row)[0])
    rul_days = rul_hours / 24

    if rul_hours <= 24:
        status = 'CRITICAL'
    elif rul_hours <= 72:
        status = 'WARNING'
    elif rul_hours <= 180:
        status = 'CAUTION'
    else:
        status = 'NORMAL'

    return {
        "machine_id": machine_id,
        "rul_hours": float(rul_hours),
        "rul_days": float(rul_days),
        "status": status,
        "machine_age_days": float(new_row.get('machine_age_days', 0)),
        "days_since_filter_change": float(
            new_row.get('days_since_filter_change', 0)
        )
    }