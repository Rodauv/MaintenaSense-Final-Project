import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

def normalizer(df, metric_columns):
    """This function normalizes the given metric column to values between 0 and 1"""
    df_normalized = df.copy()

    for column in metric_columns:
        df_normalized[column] = MinMaxScaler().fit_transform(
            np.array(df[column]).reshape(-1, 1)
        )

    return df_normalized

def standardise(df,metric_columns):
    scaler = StandardScaler()
    df_standardized = df.copy()
    df_standardized[metric_columns] = scaler.fit_transform(df[metric_columns])
    return df_standardized

# Convert to time series sequences (3D for LSTM)
def create_sequences(data, labels, timesteps):
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps:i])  # Create sequence
        y.append(labels[i])  # Keep corresponding label
    return np.array(X), np.array(y)

def test_train_split(df,cutoff):
    cutoff_date = cutoff  # Example boundary

    df_train = df[df["timestamp"] < cutoff_date]
    df_test  = df[df["timestamp"] >= cutoff_date]

    # Separate features and labels for train vs. test
    X_train = df_train.drop(columns=["timestamp","machine_status_code"])  # or your label column
    y_train = df_train["machine_status_code"]

    X_test = df_test.drop(columns=["timestamp","machine_status_code"])
    y_test = df_test["machine_status_code"]

    return X_train, X_test, y_train, y_test

def test_train_split_lstm(df, cutoff, timesteps=10):
    """
    Splits time-series data into train/test sets and reshapes it for LSTM.

    Parameters:
        df (pd.DataFrame): Full dataset with timestamp and machine_status_code.
        cutoff (str): Date boundary for train/test split.
        timesteps (int): Number of timesteps for LSTM input.

    Returns:
        X_train, X_test (numpy arrays): Scaled and reshaped datasets (3D: samples, timesteps, features)
        y_train, y_test (numpy arrays): Labels (for evaluation, not used in training)
        scaler: Fitted scaler for inverse transformation
    """
    # Split into train and test sets
    df_train = df[df["timestamp"] < cutoff].copy()
    df_test = df[df["timestamp"] >= cutoff].copy()

    # Drop timestamp and label for X
    X_train = df_train.drop(columns=["timestamp", "machine_status_code"]).values
    X_test = df_test.drop(columns=["timestamp", "machine_status_code"]).values

    y_train = df_train["machine_status_code"].values
    y_test = df_test["machine_status_code"].values

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform only (NO fit)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, timesteps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, timesteps)

    print(f"X_train shape: {X_train_seq.shape}, X_test shape: {X_test_seq.shape}")

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler

def test_train_split_dense(df, cutoff):
    """
    Splits time-series data into train/test sets for a Dense Autoencoder (2D input).

    Parameters:
        df (pd.DataFrame): Full dataset with timestamp and machine_status_code.
        cutoff (str): Date boundary for train/test split.

    Returns:
        X_train, X_test (numpy arrays): Scaled datasets (2D: samples, features)
        y_train, y_test (numpy arrays): Labels (for evaluation, not used in training)
        scaler: Fitted scaler for inverse transformation
    """
    # Split into train and test sets
    df_train = df[df["timestamp"] < cutoff].copy()
    df_test = df[df["timestamp"] >= cutoff].copy()

    # Drop timestamp and label for X
    X_train = df_train.drop(columns=["timestamp", "machine_status_code"]).values
    X_test = df_test.drop(columns=["timestamp", "machine_status_code"]).values

    y_train = df_train["machine_status_code"].values
    y_test = df_test["machine_status_code"].values

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform only (NO fit)

    print(f"X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

