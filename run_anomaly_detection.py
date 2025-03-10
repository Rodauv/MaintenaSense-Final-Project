import pandas as pd
import joblib
import numpy as np
from sqlalchemy import create_engine
from tensorflow.keras.models import load_model

# Define file paths
HISTORY_DB = "data/hist/history.sql"
NEW_READING_FILE = "data/drop/readings.csv"
GRADIENTBOOST_MODEL_PATH = "models/GradientBoost_model.pkl"
RANDOMFOREST_MODEL_PATH = "models/RandomForest_model.pkl"
AUTOENCODER_MODEL_PATH = "models/dense_autoencoder.h5"
LSTM_MODEL_PATH = "models/lstm_autoencoder.h5"
METRIC_COLUMNS = ['sensor_00', 'sensor_04', 'sensor_10', 'sensor_06', 'sensor_11', 'sensor_07', 'sensor_02']

# Connect to SQL database
engine = create_engine(f"sqlite:///{HISTORY_DB}")

def load_models():
    """Load all trained models."""
    models = {
        "GradientBoost": joblib.load(GRADIENTBOOST_MODEL_PATH),
        "RandomForest": joblib.load(RANDOMFOREST_MODEL_PATH),
        "Autoencoder": load_model(AUTOENCODER_MODEL_PATH),
        "LSTM": load_model(LSTM_MODEL_PATH)
    }
    return models

def run_anomaly_detection():
    """Runs anomaly detection on new readings and updates the database."""

    # Load historical data
    try:
        history_df = pd.read_sql("SELECT * FROM history", engine)
    except:
        history_df = pd.DataFrame(columns=["timestamp"] + METRIC_COLUMNS + ["machine_status_code"])
    
    # Load new readings
    try:
        new_data = pd.read_csv(NEW_READING_FILE)
    except FileNotFoundError:
        print("No new readings found.")
        return
    
    print(f"Loaded {len(new_data)} new readings.")

    # Ensure timestamp column exists
    if "timestamp" not in new_data.columns:
        new_data["timestamp"] = pd.to_datetime("now")
    
    # Load models
    models = load_models()

    # Run predictions (GradientBoost & RandomForest)
    new_data["prediction_gradientboost"] = models["GradientBoost"].predict(new_data[METRIC_COLUMNS])
    new_data["prediction_randomforest"] = models["RandomForest"].predict(new_data[METRIC_COLUMNS])

    # Run Autoencoder Anomaly Detection
    X_autoencoder = new_data[METRIC_COLUMNS].values
    X_reconstructed = models["Autoencoder"].predict(X_autoencoder)
    reconstruction_error = np.mean(np.square(X_autoencoder - X_reconstructed), axis=1)
    threshold_autoencoder = np.percentile(reconstruction_error, 99)
    new_data["anomaly_autoencoder"] = (reconstruction_error > threshold_autoencoder).astype(int)

    # Run LSTM Anomaly Detection
    X_lstm = new_data[METRIC_COLUMNS].values
    expected_lstm_shape = models["LSTM"].input_shape
    if len(expected_lstm_shape) == 3:
        time_steps = expected_lstm_shape[1]
        num_samples = X_lstm.shape[0] // time_steps
        X_lstm = X_lstm[: num_samples * time_steps].reshape((num_samples, time_steps, expected_lstm_shape[2]))
    
    X_reconstructed_lstm = models["LSTM"].predict(X_lstm)
    reconstruction_error_lstm = np.mean(np.square(X_lstm - X_reconstructed_lstm), axis=(1, 2))
    threshold_lstm = np.percentile(reconstruction_error_lstm, 99)
    new_data["anomaly_lstm"] = (reconstruction_error_lstm > threshold_lstm).astype(int)

    # Combine Anomalies
    new_data["anomaly_combined"] = ((new_data["anomaly_lstm"] + new_data["anomaly_autoencoder"]) >= 1).astype(int)

    # Append to SQL database
    new_data.to_sql("history", con=engine, index=False, if_exists="append")
    print(f"Updated history with {len(new_data)} new readings.")

    return new_data

if __name__ == "__main__":
    run_anomaly_detection()
