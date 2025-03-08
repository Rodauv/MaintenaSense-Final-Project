import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine
import time
from datetime import datetime

# Path to the drop folder
DROP_FOLDER = '/data/drop'
PROCESSED_FOLDER = '/data/predictions'

# Path to models
MODEL_PATH_1 = 'models/pdense_autoencoder.h5'
MODEL_PATH_2 = 'models/lstm_autoencoder.h5'

# Database connection
DATABASE_URL = 'postgresql://username:password@localhost/dbname'

# Load models
model1 = load_model(MODEL_PATH_1)
model2 = load_model(MODEL_PATH_2)

def load_and_predict(file_path):
    """
    This function loads the CSV, makes predictions using the two models,
    and appends predictions to the database.
    """
    # Load the CSV data
    df = pd.read_csv(file_path)

    # Preprocess data
    X = df.drop(columns=['target_column'])
    X = X.values

    # Model predictions
    prediction1 = model1.predict(X)
    prediction2 = model2.predict(X)

    # Add predictions to the dataframe
    df['prediction_model_1'] = prediction1
    df['prediction_model_2'] = prediction2

    # Save the predictions to the database
    save_to_database(df)

    # Move the processed file to the processed folder
    os.rename(file_path, os.path.join(PROCESSED_FOLDER, os.path.basename(file_path)))

def save_to_database(df):
    """
    This function appends the data with predictions to the database.
    """
    engine = create_engine(DATABASE_URL)
    df.to_sql('predictions_table', con=engine, if_exists='append', index=False)

def process_files():
    """
    This function processes new CSV files in the drop folder every time it's called.
    """
    # List all CSV files in the drop folder
    files = [f for f in os.listdir(DROP_FOLDER) if f.endswith('.csv')]

    if not files:
        print("No files to process.")
        return

    for file in files:
        file_path = os.path.join(DROP_FOLDER, file)
        print(f"Processing file: {file_path}")

        # Load and predict the values
        load_and_predict(file_path)

        print(f"File {file} processed successfully.")

def main():
    """
    Main function to run the process every hour, or from bash execution.
    """
    # Run the process every hour
    while True:
        process_files()
        print(f"Waiting for the next run at {datetime.now()}")
        time.sleep(3600)  # Sleep for 1 hour

if __name__ == "__main__":
    # Run the script from the command line
    main()