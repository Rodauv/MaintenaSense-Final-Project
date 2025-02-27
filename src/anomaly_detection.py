import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest

from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Dense

def detect_anomalies(df, sensors_of_interest, contamination=0.02):
    """
    ...
    """
    
    # Prepare the data for modeling (using sensor columns)
    model_data = df[sensors_of_interest].values

    # Initialize the Isolation Forest model
    model = IsolationForest(contamination=contamination)

    # Fit the model and predict anomalies
    anomalies_if = model.fit_predict(model_data)

    # Convert the prediction (-1 for anomaly, 1 for normal) to 'Y' for anomalies and 'N' for normal
    df['anomaly_isolationforest'] = [1 if x == -1 else 0 for x in anomalies_if]

    return df

def create_isolation_forest(X_train,X_test):
    # Initialize the Isolation Forest model
    model = IsolationForest(contamination=0.01, n_estimators=100, max_samples='auto', random_state=42)

    # Fit the model
    model.fit(X_train)

    # Predict anomalies on the test set
    y_pred = model.predict(X_test)

    # Convert predictions (-1 for anomaly, 1 for normal) to binary labels (0 for normal, 1 for anomaly)
    y_pred = [0 if label == 1 else 1 for label in y_pred]  # Convert to 0 for normal and 1 for anomaly

def create_autoencoder(input_dim):
    """
    ...
    """
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    #model.add(Dropout(0.2)) # drop 20% to prevent overfitting
    #model.add(Dense(32, input_dim=input_dim, activation='relu'))
    #model.add(Dropout(0.2)) # drop 20% to prevent overfitting
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))  # Latent layer with a reduced number of features
    #model.add(Dense(16, activation='relu', activity_regularizer=l1(0.001)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(input_dim, activation='sigmoid'))  # Output layer with same dimension as input
    #model.compile(optimizer='adam', loss='mean_squared_error')
    model.compile(optimizer='adam', loss='huber')
    #model.compile(optimizer='adam', loss='logcosh')
    return model

def create_lstm_autoencoder(input_shape):
    """
    Creates an LSTM-based autoencoder model for anomaly detection.
    
    Parameters:
        input_shape (tuple): Shape of the input data (timesteps, features).
        
    Returns:
        model (Keras Model): Compiled LSTM Autoencoder model.
    """
    # Encoder
    inputs = Input(shape=input_shape)
    encoded = LSTM(64, activation='relu', return_sequences=True, dropout=0.2)(inputs)
    encoded = LSTM(32, activation='relu', return_sequences=False, dropout=0.2)(encoded)  # Bottleneck

    # Decoder
    decoded = RepeatVector(input_shape[0])(encoded)  # Repeat bottleneck output for each timestep
    decoded = LSTM(32, activation='relu', return_sequences=True, dropout=0.2)(decoded)
    decoded = LSTM(64, activation='relu', return_sequences=True, dropout=0.2)(decoded)
    decoded = TimeDistributed(Dense(input_shape[1], activation='sigmoid'))(decoded)  # Reconstruct original shape

    # Compile Model
    model = Model(inputs, decoded)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model