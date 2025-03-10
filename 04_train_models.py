# Import the libraries and functions
import numpy as np
import pandas as pd
import sklearn
import tensorflow
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine

from src.data_transformations import test_train_split_dense, test_train_split_lstm, test_train_split
from src.anomaly_detection import create_autoencoder, create_lstm_autoencoder

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define input parameters
CUTOFF_DATE = "2018-06-30"
HISTORY_FILE = "data/clean/sensor_clean.pqt"
GRADIENTBOOST_MODEL_PATH = "models/GradientBoost_model.pkl"
RANDOMFOREST_MODEL_PATH = "models/RandomForest_model.pkl"
LSTM_MODEL_PATH = "models/lstm_autoencoder.h5"
AUTOENCODER_MODEL_PATH = "models/dense_autoencoder.h5"
METRIC_COLUMNS = ['sensor_00','sensor_04','sensor_06','sensor_11','sensor_02']
print("===========INPUT PARAMETERS========")
print(f"Input parameters created with cutoff date: {CUTOFF_DATE}")

# Define standard models
STANDARD_MODELS = {
    "GradientBoost": GradientBoostingClassifier(),
    "RandomForest": RandomForestClassifier()
}

# Load historical data
df = pd.read_parquet(HISTORY_FILE)
df["machine_status_code"] = df["machine_status_code"].replace({2: 1})  # Convert RECOVERING to BROKEN (1)
df = df.drop(columns=['day_of_week','hour','month'])
print("History loaded")

# Standard train-test split
X_train, X_test, y_train, y_test = test_train_split(df, CUTOFF_DATE)

# LSTM-specific train-test split
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler_lstm = test_train_split_lstm(df, CUTOFF_DATE)

# Dense Autoencoder-specific train-test split
X_train_dense, X_test_dense, y_train_dense, y_test_dense, scaler_dense = test_train_split_dense(df, CUTOFF_DATE)
print("Test & Train split complete")

# ===================== TRAIN & SAVE STANDARD MODELS ===================== #
for model_name, model in STANDARD_MODELS.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    # Save the trained model
    model_filename = f"models/{model_name}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"{model_name} saved as {model_filename}")

print("Standard Models Training Complete!")

# ===================== TRAIN & SAVE AUTOENCODER ===================== #
print("Training Dense Autoencoder...")
autoencoder = create_autoencoder(X_train_dense.shape[1])
print(f"{X_train_dense.shape=}")
autoencoder.fit(X_train_dense, X_train_dense, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
autoencoder.save(AUTOENCODER_MODEL_PATH)
print(f"Dense Autoencoder model saved to {AUTOENCODER_MODEL_PATH}")

# ===================== TRAIN & SAVE LSTM AUTOENCODER ===================== #
print("Training LSTM Autoencoder...")
lstm_model = create_lstm_autoencoder(X_train_lstm.shape[1:])
lstm_model.fit(X_train_lstm, X_train_lstm, epochs=50, batch_size=64, validation_split=0.2, verbose=1)
lstm_model.save(LSTM_MODEL_PATH)
print(f"LSTM model saved to {LSTM_MODEL_PATH}")