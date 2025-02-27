from sklearn.ensemble import IsolationForest

def detect_anomalies(df, metric_columns, contamination=0.02):
    model_data = df[metric_columns].values
    model = IsolationForest(contamination=contamination)
    
    anomalies_if = model.fit_predict(model_data)
    
    df['anomaly'] = [1 if x == -1 else 0 for x in anomalies_if]

    return df