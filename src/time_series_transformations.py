
import pandas as pd

def calculate_rolling_stats(df, time_windows, metric_columns):
    """
    Calculate rolling mean and standard deviation for specified metric columns over given time windows.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the sensor data.
    time_windows (list): A list of time windows (in minutes) for rolling calculations.
    metric_columns (list): A list of sensor column names to compute rolling stats for.
    
    Returns:
    pd.DataFrame: The original dataframe with additional rolling mean and std values labeled as "metric_column"_"time_window"_"transformation".
    """
    df = df.copy()
    df.set_index('timestamp', inplace=True)
    
    for window in time_windows:
        for sensor in metric_columns:
            df[f'{sensor}_{window}_mean'] = df[sensor].rolling(f'{window}T').mean()
            df[f'{sensor}_{window}_std'] = df[sensor].rolling(f'{window}T').std()
    
    df.reset_index(inplace=True)
    
    return df
