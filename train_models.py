# Import the libraries and functions
import numpy as np
import pandas as pd
import sklearn
import tensorflow
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine

from src.data_transformations import test_train_split_dense, test_train_split_lstm
from src.anomaly_detection import create_autoencoder, create_lstm_autoencoder

# Define input parameters
CUTOFF_DATE = "2018-06-30"
HISTORY_FILE = "data/clean/sensor_clean.pqt"
LSTM_MODEL_PATH = "models/lstm_autoencoder.h5"
AUTOENCODER_MODEL_PATH = "models/dense_autoencoder.h5"
METRIC_COLUMNS = ['sensor_00','sensor_04','sensor_10','sensor_06','sensor_11','sensor_07','sensor_02']
print("===========INPUT PARAMETERS========")
print(f"Input parameters created with custoff date: {CUTOFF_DATE}")

# Load historical data
df = pd.read_parquet(HISTORY_FILE)
print("History loaded")
# LSTM-specific train-test split
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler_lstm = test_train_split_lstm(df, CUTOFF_DATE)

# Dense Autoencoder-specific train-test split
X_train_dense, X_test_dense, y_train_dense, y_test_dense, scaler_dense = test_train_split_dense(df, CUTOFF_DATE)
print("Test & Train split complete")

# ===================== TRAIN & SAVE MODELS ===================== #
print("Training Dense Autoencoder...")
autoencoder = create_autoencoder(X_train_dense.shape[1])
print(f"{X_train_dense.shape=}")
autoencoder.fit(X_train_dense[:100], X_train_dense[:100], epochs=5, batch_size=64, validation_split=0.2, verbose=1)
autoencoder.save(AUTOENCODER_MODEL_PATH)
print(f"Dense Autoencoder model saved to {AUTOENCODER_MODEL_PATH}")

print("Training LSTM Autoencoder...")
lstm_model = create_lstm_autoencoder(X_train_lstm.shape[1:])
lstm_model.fit(X_train_lstm[:100], X_train_lstm[:100], epochs=5, batch_size=64, validation_split=0.2, verbose=1)
lstm_model.save(LSTM_MODEL_PATH)
print(f"LSTM model saved to {LSTM_MODEL_PATH}")

# ===================== TEST & EVALUATE MODELS ===================== #

# Load Autoencoder model & predict
autoencoder = load_model(AUTOENCODER_MODEL_PATH)
X_to_predict = df[METRIC_COLUMNS].values
print(f"{X_to_predict.shape=}")
X_reconstructed = autoencoder.predict(X_to_predict)
reconstruction_error = np.mean(np.square(df[METRIC_COLUMNS].values - X_reconstructed), axis=1)
threshold_autoencoder = np.percentile(reconstruction_error, 99)
df["anomaly_autoencoder"] = (reconstruction_error > threshold_autoencoder).astype(int)

# Load LSTM model & predict
lstm_model = load_model(LSTM_MODEL_PATH)
#scaler = MinMaxScaler()
#X_lstm = scaler.fit_transform(df[METRIC_COLUMNS].values)
X_lstm = df[METRIC_COLUMNS].values.reshape((df.shape[0], 10, df.shape[1] // 10))
X_reconstructed = lstm_model.predict(X_lstm)
reconstruction_error = np.mean(np.square(X_lstm - X_reconstructed), axis=(1, 2))
threshold_lstm = np.percentile(reconstruction_error, 99)
df["anomaly_lstm"] = (reconstruction_error > threshold_lstm).astype(int)

# Combine anomalies (LSTM + Autoencoder)
df["anomaly_combined"] = ((df["anomaly_lstm"] + df["anomaly_autoencoder"]) >= 1).astype(int)

# Create an SQL file
DB_FILE = "data/predictions/anomaly_results_production.sqlite"
engine = create_engine(f"sqlite:///{DB_FILE}")

# Save results
df.to_sql("anomaly_results", con=engine, index=False, if_exists="replace")
#df.to_csv("data/predictions/anomaly_results_production.csv", index=False)
print(f"Results saved to {DB_FILE}")