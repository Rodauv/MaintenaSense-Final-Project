import joblib
import numpy as np
import os
import tensorflow as tf
from azureml.core.model import Model
from sklearn.ensemble import IsolationForest

def init():
    global lstm_model, autoencoder

    # Load models from Azure ML
    iso_forest_path = Model.get_model_path("isolation_forest_final")
    lstm_model_path = Model.get_model_path("lstm_autoencoder")
    autoencoder_path = Model.get_model_path("dense_autoencoder")

    iso_forest = joblib.load(iso_forest_path)
    lstm_model = tf.keras.models.load_model(lstm_model_path)
    autoencoder = tf.keras.models.load_model(autoencoder_path)

def run(data):
    try:
        input_data = np.array(data["data"])
        
        # Isolation Forest Prediction
        iso_pred = iso_forest.predict(input_data)
        iso_anomalies = np.where(iso_pred == -1, 1, 0)

        # LSTM Prediction
        input_lstm = input_data.reshape(1, 10, input_data.shape[1] // 10)
        lstm_recon = lstm_model.predict(input_lstm)
        lstm_error = np.mean(np.square(input_lstm - lstm_recon), axis=(1, 2))
        lstm_anomalies = int(lstm_error > np.percentile(lstm_error, 99))

        # Autoencoder Prediction
        auto_recon = autoencoder.predict(input_data)
        auto_error = np.mean(np.square(input_data - auto_recon), axis=1)
        auto_anomalies = int(auto_error > np.percentile(auto_error, 99))

        # Combined Anomaly Decision
        anomaly_combined = 1 if (iso_anomalies[0] + lstm_anomalies + auto_anomalies) >= 1 else 0

        return {"isolation_forest": int(iso_anomalies[0]),
                "lstm": lstm_anomalies,
                "autoencoder": auto_anomalies,
                "anomaly_combined": anomaly_combined}

    except Exception as e:
        return str(e)
