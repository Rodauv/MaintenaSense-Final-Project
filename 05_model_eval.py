import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy import create_engine

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Define paths
GRADIENTBOOST_MODEL_PATH = "models/GradientBoost_model.pkl"
RANDOMFOREST_MODEL_PATH = "models/RandomForest_model.pkl"
AUTOENCODER_MODEL_PATH = "models/dense_autoencoder.h5"
LSTM_MODEL_PATH = "models/lstm_autoencoder.h5"

# Load trained models
gradientboost = joblib.load(GRADIENTBOOST_MODEL_PATH)
randomforest = joblib.load(RANDOMFOREST_MODEL_PATH)
autoencoder = load_model(AUTOENCODER_MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)

# Define parameters
CUTOFF_DATE = "2018-06-30"
HISTORY_FILE = "data/clean/sensor_clean.pqt"
METRIC_COLUMNS = ['sensor_00','sensor_04','sensor_10','sensor_06','sensor_11','sensor_07','sensor_02']

# Load historical data
df = pd.read_parquet(HISTORY_FILE)
df["machine_status_code"] = df["machine_status_code"].replace({2: 1})  # Convert RECOVERING to BROKEN (1)
df = df.drop(columns=['day_of_week','hour','month'])
print("History loaded")

# Standard Predictions (GradientBoost & RandomForest)
df["prediction_gradientboost"] = gradientboost.predict(df[METRIC_COLUMNS])
df["prediction_randomforest"] = randomforest.predict(df[METRIC_COLUMNS])

# ===================== AUTOENCODER ANOMALY DETECTION ===================== #
print("🔄 Running Autoencoder Anomaly Detection...")
X_autoencoder = df[METRIC_COLUMNS].values

# Ensure input shape matches autoencoder
expected_autoencoder_shape = autoencoder.input_shape
if X_autoencoder.shape[1] != expected_autoencoder_shape[1]:
    print(f"⚠️ Reshaping Autoencoder input from {X_autoencoder.shape} to match {expected_autoencoder_shape}...")
    X_autoencoder = X_autoencoder.reshape(-1, expected_autoencoder_shape[1])

# Predict and compute reconstruction error
X_reconstructed = autoencoder.predict(X_autoencoder)
reconstruction_error = np.mean(np.square(X_autoencoder - X_reconstructed), axis=1)
threshold_autoencoder = np.percentile(reconstruction_error, 99)

# Apply anomaly detection threshold
df["anomaly_autoencoder"] = (reconstruction_error > threshold_autoencoder).astype(int)
print(f"✅ Autoencoder Anomaly Detection Complete! Threshold: {threshold_autoencoder:.4f}")

# ===================== LSTM ANOMALY DETECTION ===================== #
print("🔄 Running LSTM Anomaly Detection...")

# Check expected LSTM input shape
expected_lstm_shape = lstm_model.input_shape
print(f"LSTM Model expects input shape: {expected_lstm_shape}")

X_lstm = df[METRIC_COLUMNS].values

# If LSTM was trained with sequences (time_steps)
if len(expected_lstm_shape) == 3:
    time_steps = expected_lstm_shape[1]
    features = expected_lstm_shape[2]

    print(f"⚠️ Reshaping LSTM input from {X_lstm.shape} to match (None, {time_steps}, {features})...")

    # Calculate the number of valid samples (trim excess rows)
    num_samples = X_lstm.shape[0] // time_steps
    valid_length = num_samples * time_steps

    # Trim and reshape
    X_lstm = X_lstm[:valid_length].reshape((num_samples, time_steps, features))

# Predict and compute reconstruction error
X_reconstructed_lstm = lstm_model.predict(X_lstm)

# Fix shape mismatch for reconstruction error calculation
if X_lstm.shape != X_reconstructed_lstm.shape:
    print(f"⚠️ Adjusting shapes: X_lstm {X_lstm.shape} vs X_reconstructed {X_reconstructed_lstm.shape}")
    min_samples = min(X_lstm.shape[0], X_reconstructed_lstm.shape[0])
    X_lstm = X_lstm[:min_samples]
    X_reconstructed_lstm = X_reconstructed_lstm[:min_samples]

# Compute reconstruction error for LSTM
reconstruction_error_lstm = np.mean(np.square(X_lstm - X_reconstructed_lstm), axis=(1, 2))

# 🔹 Fix the Length Mismatch
if len(reconstruction_error_lstm) < len(df):
    print(f"⚠️ Padding reconstruction_error_lstm from {len(reconstruction_error_lstm)} to match DataFrame length {len(df)}...")
    padding = np.zeros(len(df) - len(reconstruction_error_lstm))  # Fill missing values with zeros
    reconstruction_error_lstm = np.concatenate((reconstruction_error_lstm, padding))    

threshold_lstm = np.percentile(reconstruction_error_lstm, 99)

# Apply anomaly detection threshold
df["anomaly_lstm"] = (reconstruction_error_lstm > threshold_lstm).astype(int)
print(f"✅ LSTM Anomaly Detection Complete! Threshold: {threshold_lstm:.4f}")

# ===================== COMBINE ANOMALY RESULTS ===================== #
df["anomaly_combined"] = ((df["anomaly_lstm"] + df["anomaly_autoencoder"]) >= 1).astype(int)

# ===================== SAVE RESULTS TO DATABASE ===================== #
DB_FILE = "data/predictions/anomaly_results_production.sqlite"
engine = create_engine(f"sqlite:///{DB_FILE}")

# Save results
df.to_sql("anomaly_results", con=engine, index=False, if_exists="replace")
print(f"✅ Results saved to {DB_FILE}")

# ===================== SAVE MODEL PERFORMANCE METRICS ===================== #
model_performance = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

# Evaluate Standard Models
for model_name, model in [("GradientBoost", gradientboost), ("RandomForest", randomforest)]:
    y_pred = model.predict(df[METRIC_COLUMNS])

    model_performance["Model"].append(model_name)
    model_performance["Accuracy"].append(accuracy_score(df["machine_status_code"], y_pred))
    model_performance["Precision"].append(precision_score(df["machine_status_code"], y_pred, zero_division=0))
    model_performance["Recall"].append(recall_score(df["machine_status_code"], y_pred))
    model_performance["F1 Score"].append(f1_score(df["machine_status_code"], y_pred))

# Evaluate Autoencoder
model_performance["Model"].append("Autoencoder")
model_performance["Accuracy"].append(accuracy_score(df["machine_status_code"], df["anomaly_autoencoder"]))
model_performance["Precision"].append(precision_score(df["machine_status_code"], df["anomaly_autoencoder"], zero_division=0))
model_performance["Recall"].append(recall_score(df["machine_status_code"], df["anomaly_autoencoder"]))
model_performance["F1 Score"].append(f1_score(df["machine_status_code"], df["anomaly_autoencoder"]))

# Evaluate LSTM
model_performance["Model"].append("LSTM")
model_performance["Accuracy"].append(accuracy_score(df["machine_status_code"], df["anomaly_lstm"]))
model_performance["Precision"].append(precision_score(df["machine_status_code"], df["anomaly_lstm"], zero_division=0))
model_performance["Recall"].append(recall_score(df["machine_status_code"], df["anomaly_lstm"]))
model_performance["F1 Score"].append(f1_score(df["machine_status_code"], df["anomaly_lstm"]))

# Save Performance Metrics
performance_df = pd.DataFrame(model_performance)
performance_df.to_csv("models/model_performance.csv", index=False)
print("✅ Model performance metrics saved to models/model_performance.csv")