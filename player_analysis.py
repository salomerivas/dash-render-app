from sre_parse import State
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import os
import kagglehub
import plotly.express as px
import plotly.subplots as sp
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


try:
    # Download the dataset
    path = kagglehub.dataset_download("philiphyde1/nfl-stats-1999-2022")
    
    # Initialize empty DataFrames for weekly and yearly data
    week_df = pd.DataFrame()
    year_df = pd.DataFrame()

    # Iterate through the downloaded files in the dataset directory
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        
        if file_name.endswith('.csv'):
            # Identify and load specific files based on their name
            if 'weekly_player_data' in file_name:
                week_df = pd.read_csv(file_path)
            elif 'yearly_player_data' in file_name:
                year_df = pd.read_csv(file_path)
except Exception as e:
    print(f"Failed to process data: {e}")

app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])
app.title = "NFL Player Analysis"
server = app.server

# Fill missing values in critical columns
year_df.fillna({'reception_td': 0, 'run_td': 0, 'receptions': 0, 'receiving_yards': 0, 'rushing_yards': 0}, inplace=True)

# Ensure 'season' column is an integer
year_df['season'] = year_df['season'].astype(int)

# Create 'total_touchdowns' and 'total_yards' columns
year_df['total_touchdowns'] = year_df['reception_td'] + year_df['run_td']
year_df['total_yards'] = year_df['receiving_yards'] + year_df['rushing_yards']

def calculate_average_performance(df, season_range):
    # Filter data for the selected season range
    filtered_data = df[df['season'].between(season_range[0], season_range[1])]
    seasons = range(season_range[0], season_range[1] + 1)

    # Calculate averages for each season
    avg_touchdowns = (
        filtered_data.groupby('season')['total_touchdowns'].mean()
        .reindex(seasons, fill_value=0)
        .round(2)
    )
    avg_yards = (
        filtered_data.groupby('season')['total_yards'].mean()
        .reindex(seasons, fill_value=0)
        .round(2)
    )
    avg_receptions = (
        filtered_data.groupby('season')['receptions'].mean()
        .reindex(seasons, fill_value=0)
        .round(2)
    )

    return avg_touchdowns, avg_yards, avg_receptions


def generate_average_graphs(season_range):
    # Calculate averages using the helper function
    avg_touchdowns, avg_yards, avg_receptions = calculate_average_performance(year_df, season_range)

    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=3, shared_xaxes=True,
        subplot_titles=[
            "Average Total Touchdowns",
            "Average Total Yards",
            "Average Receptions"
        ]
    )

    # Add traces for averages
    seasons = list(range(season_range[0], season_range[1] + 1))
    fig.add_trace(
        go.Scatter(
            x=seasons, y=avg_touchdowns,
            mode='lines+markers', name="Avg Touchdowns"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=seasons, y=avg_yards,
            mode='lines+markers', name="Avg Yards"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=seasons, y=avg_receptions,
            mode='lines+markers', name="Avg Receptions"
        ),
        row=1, col=3
    )

    # Update layout
    fig.update_layout(
        title="Average Player Performance Metrics (Selected Seasons)",
        template="plotly_white",
        height=400
    )
    return fig



# Step 4: Define helper functions
def calculate_total_touchdowns(df, season_range):
    df = df[df['season'].between(season_range[0], season_range[1])]
    df['total_touchdowns'] = df['reception_td'] + df['run_td']
    touchdowns_per_player = df.groupby('player_name')['total_touchdowns'].sum()
    return touchdowns_per_player.sort_values(ascending=False).head(10)

def calculate_total_yardage(df, season_range):
    df = df[df['season'].between(season_range[0], season_range[1])]
    df['total_yardage'] = df['receiving_yards'] + df['rushing_yards']
    yardage_per_player = df.groupby('player_name')['total_yardage'].sum()
    return yardage_per_player.sort_values(ascending=False).head(10)

def calculate_total_receptions(df, season_range):
    df = df[df['season'].between(season_range[0], season_range[1])]
    receptions_per_player = df.groupby('player_name')['receptions'].sum()
    return receptions_per_player.sort_values(ascending=False).head(10)

def generate_bar_graph(data, title, x_label, y_label):
    players = data.index.tolist()
    values = data.values.tolist()
    colors = sample_colorscale('Pinkyl', [i / (len(players) - 1) for i in range(len(players))])
    
    fig = go.Figure(data=[
        go.Bar(
            x=players,
            y=values,
            marker=dict(color=colors),
            name=title
        )
    ])
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white'
    )
    return fig


def update_player_dropdowns(season_range):
    filtered_df = year_df[year_df['season'].between(season_range[0], season_range[1])]
    player_names = filtered_df['player_name'].dropna().unique()  # Drop NaN values
    dropdown_options = [{"label": player, "value": player} for player in player_names if player]  # Exclude empty strings
    return dropdown_options, dropdown_options, dropdown_options

def update_player_dropdowns2(season_range):
    filtered_df = year_df[year_df['season'].between(season_range[0], season_range[1])]
    player_names = filtered_df['player_name'].dropna().unique()  # Drop NaN values
    dropdown_options = [{"label": player, "value": player} for player in player_names if player]  # Exclude empty strings
    return dropdown_options, dropdown_options, dropdown_options


def update_graph(season_range, player1, player2, player3):
    # Filter data for the selected seasons
    filtered_df = year_df[year_df['season'].between(season_range[0], season_range[1])]

    # Calculate aggregated metrics for the selected players
    filtered_df['total_touchdowns'] = filtered_df['reception_td'] + filtered_df['run_td']
    filtered_df['total_yards'] = filtered_df['receiving_yards'] + filtered_df['rushing_yards']

    average_stats = (
        filtered_df.groupby('player_name')
        .agg({
            'total_touchdowns': 'mean',
            'receptions': 'mean',
            'total_yards': 'mean'
        })
        .rename(columns={
            'total_touchdowns': 'Avg Total Touchdowns',
            'receptions': 'Avg Receptions',
            'total_yards': 'Avg Total Yards'
        })
    )

    # Ensure selected players are valid
    selected_players = [player for player in [player1, player2, player3] if player]
    if not selected_players:
        return go.Figure()

    # Filter for selected players
    filtered_stats = average_stats.loc[selected_players]

    # Extract metrics for plotting
    touchdowns = filtered_stats['Avg Total Touchdowns']
    receptions = filtered_stats['Avg Receptions']
    yardage = filtered_stats['Avg Total Yards']

     # Create the comparison graph
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            "Average Total Touchdowns",
            "Average Receptions",
            "Average Total Yardage"
        ]
    )

    # Add bar charts for each metric
    fig.add_trace(
        go.Bar(
            x=filtered_stats.index,
            y=touchdowns,
            name="Avg Total Touchdowns",
            marker=dict(color="DarkOliveGreen")
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=filtered_stats.index,
            y=receptions,
            name="Avg Receptions",
            marker=dict(color="Olive")
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=filtered_stats.index,
            y=yardage,
            name="Avg Total Yardage",
            marker=dict(color="OliveDrab")
        ),
        row=1, col=3
    )

    fig.update_layout(
        title=f"Player Comparison: {season_range[0]} - {season_range[1]}",
        template="plotly_white",
        height=400
    )
    return fig


# Define helper functions for analysis and visualization
def calculate_total_touchdowns(df, season_range):
    df = df[df['season'].between(season_range[0], season_range[1])]
    df['total_touchdowns'] = df['reception_td'] + df['run_td']
    return df.groupby('player_name')['total_touchdowns'].sum().sort_values(ascending=False).head(10)


def create_wr_baseline_comparison(normalized_df, selected_players, player_colors):
    # Filter for WR and calculate the baseline
    wr_df = normalized_df[normalized_df['position'].str.upper() == 'WR']  # Ensure 'WR' matches correctly

    # Calculate total_touchdowns and total_yards for baseline
    wr_df['total_touchdowns'] = wr_df['reception_td'] + wr_df['run_td']
    wr_df['total_yards'] = wr_df['receiving_yards'] + wr_df['rushing_yards']
    wr_df['total_performance'] = (
        wr_df['total_touchdowns'] + wr_df['total_yards'] + wr_df['receptions']
    )

    # Group by t and calculate the baseline (mean values for each time step)
    baseline_df = wr_df.groupby('t').agg(total_performance=('total_performance', 'mean')).reset_index()

    # Calculate total performance for selected players
    selected_df = normalized_df[normalized_df['player_name'].isin(selected_players)]

    selected_df['total_touchdowns'] = selected_df['reception_td'] + selected_df['run_td']
    selected_df['total_yards'] = selected_df['receiving_yards'] + selected_df['rushing_yards']
    selected_df['total_performance'] = (
        selected_df['total_touchdowns'] + selected_df['total_yards'] + selected_df['receptions']
    )

    # Merge selected players with the baseline data on 't'
    comparison_df = selected_df.merge(baseline_df, on='t', suffixes=('_player', '_baseline'))

    # Create a Plotly figure
    fig = go.Figure()

    # Plot baseline total performance
    fig.add_trace(go.Scatter(
        x=baseline_df['t'],
        y=baseline_df['total_performance'],
        mode='lines',
        name='Baseline Total Performance',
        line=dict(color='gray', dash='dash'),
        legendgroup='baseline'
    ))

    # Plot each player's performance
    for player_name in selected_players:
        player_data = comparison_df[comparison_df['player_name'] == player_name]
        fig.add_trace(go.Scatter(
            x=player_data['t'],
            y=player_data['total_performance_player'],
            mode='lines+markers',
            name=f"Total Performance: {player_name}",
            line=dict(color=player_colors[player_name]),
            marker=dict(size=8)
        ))

    # Update layout for the figure
    fig.update_layout(
        title="Comparison of Selected Players Against WR Baseline",
        xaxis_title="Normalized Time (t)",
        yaxis_title="Performance Metrics",
        template='plotly_white',
        legend_title='Players',
        hovermode='closest',
        height=600
    )

    return fig


def normalize_seasons(player_data):
    player_data = player_data.sort_values('season')  # Ensure sorted by season
    player_data['t'] = range(len(player_data))  # Create t=0, t=1, etc.
    return player_data

# Apply normalization
normalized_df = year_df.groupby('player_name', group_keys=False).apply(normalize_seasons)



def generate_bar_graph(data, title, x_label, y_label):
    players = data.index.tolist()
    values = data.values.tolist()
    colors = sample_colorscale('Pinkyl', [i / (len(players) - 1) for i in range(len(players))])
    fig = go.Figure(data=[go.Bar(x=players, y=values, marker=dict(color=colors), name=title)])
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template='plotly_white')
    return fig

def plot_seaborn_weighted_scores(players, rf_predictions, seasons_played):
    max_seasons = max(seasons_played)
    weights = [max_seasons / sp for sp in seasons_played]
    weighted_rf_predictions = rf_predictions * weights

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.bar(players, weighted_rf_predictions, color=sns.color_palette("Greens", len(players)), alpha=0.8)
    plt.title("Weighted Sponsorship Scores - Random Forest", fontsize=14)
    plt.xlabel("Player", fontsize=12)
    plt.ylabel("Weighted Sponsorship Score", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(min(weighted_rf_predictions) * 0.95, max(weighted_rf_predictions) * 1.05)

    for i, score in enumerate(weighted_rf_predictions):
        plt.text(i, score + 10, f"{score:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    # Save plot as an image
    plt.savefig("weighted_sponsorship_scores.png")
    plt.close()  # Close the plot to free memory

# Plotly Visualization (Dash Integration)
def create_plotly_weighted_scores(players, rf_predictions, seasons_played):
    max_seasons = max(seasons_played)
    weights = [max_seasons / sp for sp in seasons_played]
    weighted_rf_predictions = rf_predictions * weights

    colors = px.colors.sequential.Aggrnyl

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=players,
            y=weighted_rf_predictions,
            marker=dict(color=colors[:len(players)]),
            text=[f"{score:.1f}" for score in weighted_rf_predictions],
            textposition="outside",
            name="Weighted Sponsorship Score"
        )
    )
    fig.update_layout(
        title="Weighted Sponsorship Scores - Random Forest",
        xaxis_title="Player",
        yaxis_title="Weighted Sponsorship Score",
        template="plotly_white",
        height=600,
        bargap=0.3
    )
    return fig

# Initialize and train rf_model
def initialize_rf_model():
    # Prepare training data
    training_data = year_df.copy()
    training_data = training_data[[
        "player_name", "age", "receptions", "receiving_yards",
        "rushing_yards", "reception_td", "run_td"
    ]]
    training_data.rename(columns={
        "receptions": "total_receptions",
        "receiving_yards": "total_receiving_yards",
        "rushing_yards": "total_rushing_yards",
        "reception_td": "total_reception_td",
        "run_td": "total_run_td"
    }, inplace=True)

    # Add suitability score for training
    training_data["suitability_score"] = (
        training_data["total_receptions"] * 0.3 +
        training_data["total_receiving_yards"] * 0.4 +
        training_data["total_rushing_yards"] * 0.1 +
        training_data["total_reception_td"] * 0.15 +
        training_data["total_run_td"] * 0.05
    )

    # Prepare features and target
    X = training_data[[
        "age", "total_receptions", "total_receiving_yards",
        "total_rushing_yards", "total_reception_td", "total_run_td"
    ]]
    y = training_data["suitability_score"]

    # Train the Random Forest model
    rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_model.fit(X, y)
    return rf_model


# Train rf_model
rf_model = initialize_rf_model()



def train_models(df, rf_model=None):
    """
    Train or use pre-trained models to make predictions.
    """
    # Ensure required columns are present
    required_columns = [
        "age", "total_receptions", "total_receiving_yards",
        "total_rushing_yards", "total_reception_td", "total_run_td", "player_name"
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate the composite suitability score
    df["suitability_score"] = (
        df["total_receptions"] * 0.2 +
        df["total_receiving_yards"] * 0.2 +
        df["total_rushing_yards"] * 0.2 +
        df["total_reception_td"] * 0.2 +
        df["total_run_td"] * 0.2
    )

    # Features and target
    X = df[[
        "age", "total_receptions", "total_receiving_yards",
        "total_rushing_yards", "total_reception_td", "total_run_td"
    ]]
    y = df["suitability_score"]

    if rf_model is None:
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Random Forest model
        rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)

    # Make predictions
    df["rf_predictions"] = rf_model.predict(X)

    return {"rf_model": rf_model, "data": df}




# Step 5: Create the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "NFL Player Analysis"

color_palette = px.colors.qualitative.Dark24

# Layout for the page
player_layout = html.Div([
   html.H1("NFL Player Performance Analysis", style={"textAlign": "center", "marginBottom": "20px"}),

    # Slider for selecting seasons
    html.Div([
        html.Label("Select Seasons:"),
        dcc.RangeSlider(
            id="season-slider",
            min=year_df['season'].min(),
            max=year_df['season'].max(),
            step=1,
            marks={year: str(year) for year in range(year_df['season'].min(), year_df['season'].max() + 1)},
            value=[2020, 2023],  # Default range
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={"margin": "20px"}),

    # Graph container for averages
    dcc.Graph(id="averages-graph", style={"marginTop": "20px"}),

    # Graph containers for individual player metrics
    dcc.Graph(id="touchdowns-graph", style={"marginTop": "20px"}),
    dcc.Graph(id="yardage-graph", style={"marginTop": "20px"}),
    dcc.Graph(id="receptions-graph", style={"marginTop": "20px"}),

    # Player Comparison Section
    html.H1("Player Comparison with Dynamic Season Selection", style={"textAlign": "center", "marginTop": "40px"}),

    # Dropdowns for player selection
    html.Div([
        html.Div([
            html.Label("Select Player 1:"),
            dcc.Dropdown(id="player1-dropdown", style={"width": "130%"})
        ], style={"display": "inline-block", "margin": "10px"}),

        html.Div([
            html.Label("Select Player 2:"),
            dcc.Dropdown(id="player2-dropdown", style={"width": "130%"})
        ], style={"display": "inline-block", "margin": "10px"}),

        html.Div([
            html.Label("Select Player 3:"),
            dcc.Dropdown(id="player3-dropdown", style={"width": "130%"})
        ], style={"display": "inline-block", "margin": "10px"})
    ]),

    # Graph to display the comparison
    dcc.Graph(id="comparison-graph", style={"marginTop": "20px"}),

    html.H1("Player Comparison with Baseline", style={"textAlign": "center", "marginBottom": "20px"}),

    # Dropdown for selecting players
    html.Div([
        html.Label("Select Players for Comparison:"),
        dcc.Dropdown(
            id="players-dropdown",
            options=[],  # Options will be populated dynamically
            multi=True,
            value=[],
            placeholder="Select up to 3 players",
            style={"width": "60%", "margin": "0 auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "20px"}),

    # Graph to display the baseline comparison
    dcc.Graph(id="baseline-comparison-graph", style={"marginTop": "20px"}),

   html.Div([
    html.H3("Player Sponsorship Prediction", style={"textAlign": "center"}),
    html.Div([
        html.Div([
            html.Label("Select Player 1:"),
            dcc.Dropdown(id="player1-1-dropdown", style={"width": "100%"})
        ], style={"display": "inline-block", "width": "30%", "padding": "10px"}),

        html.Div([
            html.Label("Select Player 2:"),
            dcc.Dropdown(id="player2-1-dropdown", style={"width": "100%"})
        ], style={"display": "inline-block", "width": "30%", "padding": "10px"}),

        html.Div([
            html.Label("Select Player 3:"),
            dcc.Dropdown(id="player3-1-dropdown", style={"width": "100%"})
        ], style={"display": "inline-block", "width": "30%", "padding": "10px"})
    ], style={"textAlign": "center"}),

    html.Div([
        dcc.Graph(id="player-metrics-visualization")
    ], style={"marginTop": "30px"})
])
])

@app.callback(
    [
        Output("touchdowns-graph", "figure"),
        Output("yardage-graph", "figure"),
        Output("receptions-graph", "figure"),
        Output("averages-graph", "figure"),
    ],
    [Input("season-slider", "value")]
)
def update_all_graphs(season_range):
    # Generate bar graphs for touchdowns, yardage, and receptions
    touchdowns_data = calculate_total_touchdowns(year_df, season_range)
    yardage_data = calculate_total_yardage(year_df, season_range)
    receptions_data = calculate_total_receptions(year_df, season_range)

    touchdowns_fig = generate_bar_graph(
        touchdowns_data,
        title="Top 10 Touchdown Leaders (Selected Seasons)",
        x_label="Player",
        y_label="Touchdowns"
    )

    yardage_fig = generate_bar_graph(
        yardage_data,
        title="Top 10 Yardage Leaders (Selected Seasons)",
        x_label="Player",
        y_label="Yardage"
    )

    receptions_fig = generate_bar_graph(
        receptions_data,
        title="Top 10 Receptions Leaders (Selected Seasons)",
        x_label="Player",
        y_label="Receptions"
    )

    # Generate the averages graph
    averages_fig = generate_average_graphs(season_range)

    return touchdowns_fig, yardage_fig, receptions_fig, averages_fig

@app.callback(
    [Output("player1-dropdown", "options"),
     Output("player2-dropdown", "options"),
     Output("player3-dropdown", "options")],
    [Input("season-slider", "value")]
)
def update_player_dropdowns_callback(season_range):
    return update_player_dropdowns(season_range)

@app.callback(
    [Output("player1-1-dropdown", "options"),
     Output("player2-1-dropdown", "options"),
     Output("player3-1-dropdown", "options")],
    [Input("season-slider", "value")]
)
def update_dropdown_options(season_range):
    filtered_df = year_df[year_df["season"].between(season_range[0], season_range[1])]
    players = filtered_df["player_name"].dropna().unique()
    options = [{"label": player, "value": player} for player in sorted(players)]
    return options, options, options


@app.callback(
    Output("comparison-graph", "figure"),
    [
        Input("season-slider", "value"),
        Input("player1-dropdown", "value"),
        Input("player2-dropdown", "value"),
        Input("player3-dropdown", "value")
    ]
)
def update_comparison_graph(season_range, player1, player2, player3):
    return update_graph(season_range, player1, player2, player3)

@app.callback(
    [Output("players-dropdown", "options"), 
     Output("baseline-comparison-graph", "figure")],
    [Input("players-dropdown", "value")]
)
def update_dropdown_and_graph(selected_players):
    # Populate dropdown options with unique player names
    options = [{"label": player, "value": player} for player in sorted(normalized_df['player_name'].unique())]

    # If no players are selected, return default options and an empty graph
    if not selected_players or len(selected_players) > 3:
        return options, go.Figure()

    # Assign dynamic colors to the selected players
    player_colors = {
        player: color_palette[i % len(color_palette)]
        for i, player in enumerate(selected_players)
    }

    # Create and return the baseline comparison graph
    baseline_graph = create_wr_baseline_comparison(normalized_df, selected_players, player_colors)
    return options, baseline_graph

@app.callback(
    Output("player-metrics-visualization", "figure"),
    [Input("player1-1-dropdown", "value"),
     Input("player2-1-dropdown", "value"),
     Input("player3-1-dropdown", "value")]
)
def update_weighted_graph(player1, player2, player3):
    # Ensure valid player selections
    selected_players = [player for player in [player1, player2, player3] if player]
    if not selected_players:
        return go.Figure()  # Return empty figure if no players are selected

    # Filter the data for selected players
    filtered_df = year_df[year_df["player_name"].isin(selected_players)]
    if filtered_df.empty:
        raise ValueError("No data available for the selected players.")

    # Compute metrics for each selected player
    metrics = filtered_df.groupby("player_name").agg({
        "total_receptions": "sum",
        "total_receiving_yards": "sum",
        "total_rushing_yards": "sum",
        "total_touchdowns": "sum",
        "age": "mean"
    }).reset_index()

    # Add composite score (suitability_score)
    metrics["suitability_score"] = (
        metrics["total_receptions"] * 0.2 +
        metrics["total_receiving_yards"] * 0.2 +
        metrics["total_rushing_yards"] * 0.2 +
        metrics["total_touchdowns"] * 0.2 +
        metrics["age"] * 0.2
    )

    # Predict using the pre-trained Random Forest model
    X = metrics[["age", "total_receptions", "total_receiving_yards", "total_rushing_yards", "total_touchdowns"]]
    metrics["rf_predictions"] = rf_model.predict(X)

    # Calculate weighted scores
    max_seasons = max(year_df.groupby("player_name")["season"].nunique())
    metrics["seasons_played"] = year_df.groupby("player_name")["season"].nunique().reindex(metrics["player_name"]).fillna(1)
    metrics["weights"] = max_seasons / metrics["seasons_played"]
    metrics["weighted_score"] = metrics["rf_predictions"] * metrics["weights"]

    # Generate the bar graph
    fig = go.Figure(
        data=[
            go.Bar(
                x=metrics["player_name"],
                y=metrics["weighted_score"],
                text=[f"{score:.2f}" for score in metrics["weighted_score"]],
                textposition="outside",
                marker_color=px.colors.sequential.Aggrnyl[:len(metrics)]
            )
        ]
    )

    fig.update_layout(
        title="Weighted Sponsorship Scores",
        xaxis_title="Player",
        yaxis_title="Weighted Score",
        template="plotly_white"
    )

    return fig





app.layout = player_layout

if __name__ == "__main__":
    app.run_server(debug=False)
