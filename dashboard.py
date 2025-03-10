import dash
from dash import dcc, html, Output, Input, ctx
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import os

# Import the anomaly detection function
from run_anomaly_detection import run_anomaly_detection

# Define database path
HISTORY_DB = "data/hist/history.sql"
engine = create_engine(f"sqlite:///{HISTORY_DB}")

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Anomaly Detection Dashboard"),
    
    # Run Anomaly Detection Button
    html.Button("Run Anomaly Detection", id="run-anomaly-btn", n_clicks=0, style={"margin-bottom": "20px"}),
    
    # Placeholder for confirmation message
    html.Div(id="run-anomaly-output", style={"font-weight": "bold", "color": "green"}),

    # Latest Readings Table
    html.H3("Latest Readings"),
    dcc.Interval(id="interval-component", interval=60000, n_intervals=0),  # Auto-refresh every minute
    html.Div(id="latest-readings"),

    # Anomaly Trends Graph
    html.H3("Anomalies Over Time"),
    dcc.Graph(id="anomaly-trend"),

    # Detected Anomalies Table
    html.H3("Detected Anomalies"),
    html.Div(id="anomalies-table")
])

# Function to Load Data
def load_data():
    if os.path.exists(HISTORY_DB):
        df = pd.read_sql("SELECT * FROM history", engine)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    else:
        return pd.DataFrame(columns=["timestamp", "anomaly_combined"])

# Callback to Update the Dashboard Automatically
@app.callback(
    Output("latest-readings", "children"),
    Output("anomaly-trend", "figure"),
    Output("anomalies-table", "children"),
    Input("interval-component", "n_intervals")
)
def update_dashboard(_):
    df = load_data()

    # Latest Readings
    latest_readings_table = html.Table([
        html.Tr([html.Th(col) for col in df.columns]),
        *[html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(10, len(df)))]
    ])

    # Anomalies Over Time Graph
    df["date"] = df["timestamp"].dt.date
    anomaly_counts = df.groupby("date")["anomaly_combined"].sum()
    anomaly_fig = px.line(anomaly_counts, x=anomaly_counts.index, y=anomaly_counts.values, 
                          labels={"x": "Date", "y": "Number of Anomalies"}, title="Anomaly Trends")

    # Detected Anomalies Table
    anomalies_df = df[df["anomaly_combined"] == 1]
    anomalies_table = html.Table([
        html.Tr([html.Th(col) for col in anomalies_df.columns]),
        *[html.Tr([html.Td(anomalies_df.iloc[i][col]) for col in anomalies_df.columns]) for i in range(min(10, len(anomalies_df)))]
    ])

    return latest_readings_table, anomaly_fig, anomalies_table

# Callback to Run Anomaly Detection
@app.callback(
    Output("run-anomaly-output", "children"),
    Input("run-anomaly-btn", "n_clicks")
)
def run_anomaly_detection_callback(n_clicks):
    if n_clicks > 0:
        ctx.triggered_id
        run_anomaly_detection()  # Run the function
        return "    Anomaly Detection Completed & Database Updated!"
    return ""

# Run Dash App
if __name__ == "__main__":
    app.run_server(debug=True)
