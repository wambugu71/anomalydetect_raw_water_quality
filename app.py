import threading
import paho.mqtt.client as mqtt
import json
import time
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import dash_daq as daq
import dash_bootstrap_components as dbc

# Set the default template for all plotly figures
pio.templates.default = "plotly_dark"

# Initialize MQTT data container for turbidity
shared_data = {'turbidity': 0.0}  # Default value for turbidity

# MQTT broker settings
mqtt_broker = "test.mosquitto.org"
mqtt_port = 1883
mqtt_topic = "ken/data"

# MQTT callback functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT broker with code {rc}")
        client.subscribe(mqtt_topic)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())  # Parse JSON payload
        new_turbidity = payload['datajson'][0]
        shared_data['turbidity'] = float(new_turbidity)  # Update shared turbidity data
        print(f"Turbidity updated: {new_turbidity}")  # Debug print
    except json.JSONDecodeError:
        print("Invalid JSON format")

# MQTT loop function (runs in a separate thread)
def mqtt_function():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(mqtt_broker, mqtt_port, 60)
    client.loop_forever()

# Start the MQTT thread
mqtt_thread = threading.Thread(target=mqtt_function)
mqtt_thread.daemon = True  # Daemonize the thread so it exits with the main program
mqtt_thread.start()

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment

# Custom CSS for additional styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Water Quality Anomaly Detection</title>
        {%favicon%}
        {%css%}
        <style>
    body {
        background-color: #f8f9fa;
        color: #333333;
    }
    .card {
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    .nav-link:hover {
        background-color: rgba(0, 0, 0, 0.05);
    }
</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Store data for plotting
data_store = {'time': [], 'turbidity': [], 'pH': []}

# Generate synthetic data for pH
def generate_data():
    current_time = pd.Timestamp.now()
    turbidity = shared_data['turbidity']  # Get real turbidity from MQTT
    pH = np.random.normal(7, 0.2)  # Random pH value
    return {'time': current_time, 'turbidity': turbidity, 'pH': pH}

# Uptime data
def uptime_data(df):
    new_data = pd.DataFrame([generate_data()])
    return pd.concat([df, new_data], ignore_index=True)

# Train Isolation Forest for anomaly detection and scoring
def train_isolation_forest(df, contamination, n_estimators, max_samples):
    X = df[['turbidity', 'pH']]
    hyper_params = {
        'bootstrap': False,
        'contamination': contamination,
        'max_features': 1.0,
        'max_samples': max_samples,
        'n_estimators': n_estimators
    }
    model = IsolationForest(**hyper_params)
    model.fit(X)
    df['anomaly'] = model.predict(X)
    df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)  # Mark anomalies
    df['score'] = -model.decision_function(X)  # Higher score indicates more anomalous
    
    return model, df

# Function to get arrow colors based on metric changes
def get_arrow_color(previous, current):
    if current > previous:
        return 'red', '↑'
    elif current < previous:
        return 'green', '↓'
    else:
        return 'gray', '→'

# Initialize dataframe
df = pd.DataFrame(columns=['time', 'turbidity', 'pH', 'anomaly'])
model = None
previous_turbidity = None
previous_ph = None

# Navbar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="#")),
        dbc.NavItem(dbc.NavLink("About", href="#")),
        dbc.NavItem(dbc.NavLink("Contact", href="#")),
    ],
    brand="Real-Time Anomaly Detection in Water Quality",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4",
)

# Sidebar for model controls
sidebar = dbc.Card([
    dbc.CardHeader(html.H4("Model Parameters", className="text-center")),
    dbc.CardBody([
        html.P("Adjust the model parameters below:"),
        dbc.Form([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Contamination"),
                    dcc.Slider(
                        id='contamination-slider',
                        min=0.01, max=0.5, step=0.01, value=0.05,
                        marks={i/10: str(i/10) for i in range(1, 6)},
                    ),
                ]),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Number of Estimators"),
                    dcc.Slider(
                        id='n-estimators-slider',
                        min=10, max=200, step=10, value=50,
                        marks={i: str(i) for i in range(10, 201, 50)},
                    ),
                ]),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Max Samples"),
                    dcc.Slider(
                        id='max-samples-slider',
                        min=10, max=1000, step=10, value=100,
                        marks={i: str(i) for i in range(10, 1001, 250)},
                    ),
                ]),
            ], className="mb-3"),
        ]),
    ]),
], className="mb-4")

# Layout of the app
app.layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Row([
            dbc.Col(sidebar, width=3),
            dbc.Col([
                #html.H1("Real-Time Anomaly Detection in Water Quality", className="text-center mb-4"),
                
                # Card for Turbidity
                dbc.Card([
                    dbc.CardHeader(html.H4("Turbidity", className="text-center")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(id='turbidity-card', className="text-center h3 mb-3"),
                                daq.Gauge(
                                    id='turbidity-gauge',
                                    min=0,
                                    max=500,
                                    value=5,
                                    showCurrentValue=True,
                                    units="NTU",
                                    color={"gradient": True, "ranges": {"green": [0, 100], "yellow": [100, 200], "red": [200, 500]}},
                                ),
                            ], width=3),
                            dbc.Col([
                                dcc.Graph(id='turbidity-graph')
                            ], width=9)
                        ]),
                    ]),
                ], className="mb-4"),
                
                # Card for pH
                dbc.Card([
                    dbc.CardHeader(html.H4("pH", className="text-center")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(id='ph-card', className="text-center h3 mb-3"),
                                daq.Gauge(
                                    id='ph-gauge',
                                    min=0,
                                    max=14,
                                    value=7,
                                    showCurrentValue=True,
                                    units="pH",
                                    color={"gradient": True, "ranges": {"green": [6, 8], "red": [4, 6], "blue": [0, 4]}},
                                ),
                            ], width=3),
                            dbc.Col([
                                dcc.Graph(id='ph-graph')
                            ], width=9),
                        ]),
                    ]),
                ]),

                # New card for Summary Statistics
                dbc.Card([
                    dbc.CardHeader(html.H4("Summary Statistics", className="text-center")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col(html.Div(id='summary-stats'), width=12),
                        ]),
                    ]),
                ], className="mt-4"),
            ], width=9)
        ])
    ], fluid=True),
    
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # 1 second interval
        n_intervals=0
    )
])

# Callback to update graphs, cards, and gauges
@app.callback(
    [Output('turbidity-graph', 'figure'),
     Output('ph-graph', 'figure'),
     Output('turbidity-card', 'children'),
     Output('turbidity-card', 'style'),
     Output('ph-card', 'children'),
     Output('ph-card', 'style'),
     Output('turbidity-gauge', 'value'),
     Output('ph-gauge', 'value'),
     #Output('turbidity-score-bar', 'value'),
     #Output('ph-score-bar', 'value'),
     Output('summary-stats', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('contamination-slider', 'value'),
     Input('n-estimators-slider', 'value'),
     Input('max-samples-slider', 'value')]
)
def update_graphs_and_cards(n, contamination, n_estimators, max_samples):
    global df, model, previous_turbidity, previous_ph

    # Uptime data and retrain model
    df = uptime_data(df)
    model, df = train_isolation_forest(df, contamination, n_estimators, max_samples)

    # Get latest values
    latest_turbidity = df['turbidity'].iloc[-1]
    latest_ph = df['pH'].iloc[-1]

    # Set previous values if none
    if previous_turbidity is None:
        previous_turbidity = latest_turbidity
    if previous_ph is None:
        previous_ph = latest_ph

    # Get color and direction based on metric changes
    turbidity_color, turbidity_status = get_arrow_color(previous_turbidity, latest_turbidity)
    ph_color, ph_status = get_arrow_color(previous_ph, latest_ph)

    # Update previous values
    previous_turbidity = latest_turbidity
    previous_ph = latest_ph

    # Filter anomalies for marking X
    anomalies = df[df['anomaly'] == 1]

    # Turbidity figure with scoring axis
    fig_turbidity = go.Figure()
    fig_turbidity.add_trace(go.Scatter(x=df['time'], y=df['turbidity'], mode='lines', name='Turbidity', line=dict(color='#3498db')))
    fig_turbidity.add_trace(go.Scatter(x=anomalies['time'], y=anomalies['turbidity'], mode='markers',
                                       marker=dict(color='red', symbol='x', size=10), name='Turbidity Anomaly'))
    fig_turbidity.add_trace(go.Scatter(x=df['time'], y=df['score'], mode='lines', name='Score', yaxis='y2', line=dict(color='#f39c12', dash='dot')))
    fig_turbidity.update_layout(
        title="Turbidity Trend",
        xaxis_title="Time",
        yaxis_title="Turbidity",
        showlegend=True,
        yaxis2=dict(
            title="Score",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # ... [Previous code remains unchanged] ...

    # pH figure with scoring axis
    fig_ph = go.Figure()
    fig_ph.add_trace(go.Scatter(x=df['time'], y=df['pH'], mode='lines', name='pH', line=dict(color='#2ecc71')))
    fig_ph.add_trace(go.Scatter(x=anomalies['time'], y=anomalies['pH'], mode='markers',
                                marker=dict(color='red', symbol='x', size=10), name='pH Anomaly'))
    fig_ph.add_trace(go.Scatter(x=df['time'], y=df['score'], mode='lines', name='Score', yaxis='y2', line=dict(color='#f39c12', dash='dot')))
    fig_ph.update_layout(
        title="pH Trend",
        xaxis_title="Time",
        yaxis_title="pH",
        showlegend=True,
        yaxis2=dict(
            title="Score",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Update cards and gauges
    turbidity_card = f"Turbidity: {latest_turbidity:.2f} NTU {turbidity_status}"
    turbidity_card_style = {'color': turbidity_color}
    ph_card = f"pH: {latest_ph:.2f} {ph_status}"
    ph_card_style = {'color': ph_color}

    # Update score bars
    #latest_turbidity_score = df['score'].iloc[-1]
    #latest_ph_score = df['score'].iloc[-1]

    # Summary statistics
    summary_stats = html.Div([
        html.H5("Turbidity Statistics:"),
        html.P(f"Mean: {df['turbidity'].mean():.2f} NTU"),
        html.P(f"Std Dev: {df['turbidity'].std():.2f} NTU"),
        html.P(f"Anomalies: {df[df['anomaly'] == 1]['turbidity'].count()}"),
        html.H5("pH Statistics:"),
        html.P(f"Mean: {df['pH'].mean():.2f}"),
        html.P(f"Std Dev: {df['pH'].std():.2f}"),
        html.P(f"Anomalies: {df[df['anomaly'] == 1]['pH'].count()}"),
    ])

    return (fig_turbidity, fig_ph, turbidity_card, turbidity_card_style, ph_card, ph_card_style,
            latest_turbidity, latest_ph, summary_stats)# latest_turbidity_score, latest_ph_score,

if __name__ == '__main__':
    app.run_server(debug=True)