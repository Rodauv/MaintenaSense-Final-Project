import matplotlib.pyplot as plt
import seaborn as sns

def time_series_error_by_sensor(df,df_broken,df_recovery,metric_columns): 
    """
    Plot time series error by sensor.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing sensor data.
    df_broken (pd.DataFrame): The dataframe containing timestamps of breakdown events.
    df_recovery (pd.DataFrame): The dataframe containing timestamps of recovery events.
    metric_columns (list): The sensor columns to plot.
    """

    fig1 = plt.figure(figsize=(30, 40))
    grid = fig1.add_gridspec(7, 1, hspace=0.5, wspace=0.3)

    columns = metric_columns

    for i, col in enumerate(columns):
        row = i
        colpos = 0
        ax = fig1.add_subplot(grid[row, colpos])


        # Plot normal data in blue line
        ax.plot(
            df["timestamp"],
            df[col],
            color="blue",
            label="Normal"
        )
        # Plot recovering data in green "X"
        ax.plot(
            df_recovery["timestamp"],
            df_recovery[col],
            linestyle="none",
            marker="X",
            color="green",
            markersize=6,
            label="Recovering"
        )
        # Plot broken data in red "X"
        ax.plot(
            df_broken["timestamp"],
            df_broken[col],
            linestyle="none",
            marker="X",
            color="red",
            markersize=6,
            label="Broken"
        )
        ax.set_title(col)
        ax.legend()

    plt.show()

def time_window_rolling_stats(df, df_broken, metric_columns, time_windows):
    """
    Plot rolling mean and standard deviation of sensor data along with breakdown events.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing original and rolling sensor data.
    df_broken (pd.DataFrame): The dataframe containing timestamps of breakdown events.
    metric_columns (list): The sensor columns to plot.
    time_windows (list): The list of time windows to use for plotting.
    
    """
    fig, axs = plt.subplots(len(metric_columns), len(time_windows), figsize=(70, 50))
    
    for i, sensor in enumerate(metric_columns):
        for j, window in enumerate(time_windows):
            mean_col = f'{sensor}_{window}_mean'
            std_col = f'{sensor}_{window}_std'
            
            if mean_col in df.columns and std_col in df.columns:
                axs[i, j].plot(df['timestamp'], df[mean_col], label=f'{sensor} Mean', color='blue')
                axs[i, j].plot(df['timestamp'], df[std_col], label=f'{sensor} Std Dev', color='orange')
                
                # Overlay the breakdown events (highlight 'BROKEN' timestamps)
                broken_timestamps = df_broken['timestamp']
                axs[i, j].scatter(broken_timestamps, df.loc[df['timestamp'].isin(broken_timestamps), mean_col], 
                                  color='red', label='BROKEN', marker='x', s=50)
                
                axs[i, j].set_title(f'{sensor} - {window} min window')
                axs[i, j].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def show_anomaly_distribution(df,anomalies_df,breakdowns_df):
    # Plot the distribution of anomalies around errors (breakdowns)
    plt.figure(figsize=(12, 6))

    # Plot breakdowns as vertical lines (e.g., at the breakdown timestamps)
    for breakdown in breakdowns_df['timestamp']:
        plt.axvline(x=breakdown, color='red', linestyle='--', label=breakdown, alpha=0.7)

    # Plot the anomalies (distribution of anomalies' timestamps)
    sns.histplot(anomalies_df['timestamp'], kde=True, color='blue', label='Anomalies', bins=50)

    # Adding labels and title
    plt.xlabel('Timestamp')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomalies Around Errors (Breakdowns)')
    plt.legend()

    plt.tight_layout()
    plt.show()