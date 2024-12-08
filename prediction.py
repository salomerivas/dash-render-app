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



    # Define the seasons for which averages will be calculated
seasons = [2020, 2021, 2022, 2023]

# Initialize lists to store the averages
average_touchdowns_per_season = []
average_yards_per_season = []
average_receptions_per_season = []

# Ensure necessary columns are calculated
year_df['total_touchdowns'] = year_df['reception_td'] + year_df['run_td']
year_df['total_yards'] = year_df['receiving_yards'] + year_df['rushing_yards']

# Calculate averages for each season
for season in seasons:
    # Filter the DataFrame for the specific season
    season_data = year_df[year_df['season'] == season]

    # Calculate averages for touchdowns, yards, and receptions
    avg_touchdowns = round(season_data['total_touchdowns'].mean(), 2)
    avg_yards = round(season_data['total_yards'].mean(), 2)
    avg_receptions = round(season_data['receptions'].mean(), 2)

    # Append the averages to the respective lists
    average_touchdowns_per_season.append(avg_touchdowns)
    average_yards_per_season.append(avg_yards)
    average_receptions_per_season.append(avg_receptions)


def create_first_graph():
    fig = sp.make_subplots(rows=1, cols=3, shared_xaxes=True, subplot_titles=(
        "Average Total Touchdowns (2020-2023)",
        "Average Total Yards (2020-2023)",
        "Average Receptions (2020-2023)"
    ))

    # Add the Average Total Touchdowns plot
    fig.add_trace(go.Scatter(
        x=seasons,
        y=average_touchdowns_per_season,
        mode='lines+markers',
        name="Average Total Touchdowns",
        line=dict(color='DarkOliveGreen')
    ), row=1, col=1)

    # Add the Average Total Yards plot
    fig.add_trace(go.Scatter(
        x=seasons,
        y=average_yards_per_season,
        mode='lines+markers',
        name="Average Total Yards",
        line=dict(color='DarkOliveGreen')
    ), row=1, col=2)

    # Add the Average Receptions plot
    fig.add_trace(go.Scatter(
        x=seasons,
        y=average_receptions_per_season,
        mode='lines+markers',
        name="Average Receptions",
        line=dict(color='DarkOliveGreen')
    ), row=1, col=3)

    # Update layout
    fig.update_layout(
        title="Player Performance Averages by Season (2020-2023)",
        template="plotly_white",
        showlegend=False,  # Turn off legend to avoid repetition
        height=400,
        width=1200
    )

    # Update x-axis titles for each subplot
    fig.update_xaxes(title_text="Seasons", row=1, col=1)
    fig.update_xaxes(title_text="Seasons", row=1, col=2)
    fig.update_xaxes(title_text="Seasons", row=1, col=3)

    # Update y-axis titles for each subplot
    fig.update_yaxes(title_text="Average Touchdowns", row=1, col=1)
    fig.update_yaxes(title_text="Average Yards", row=1, col=2)
    fig.update_yaxes(title_text="Average Receptions", row=1, col=3)

    return fig   




# New player data
touchdown_df = pd.DataFrame({
    "Player": [
        "Austin Ekeler", "Christian McCaffrey", "Davante Adams",
        "Derrick Henry", "Ezekiel Elliot", "Jalen Hurts",
        "Jonathan Taylor", "Mike Evans", "Tyreek Hill"
    ],
    "Total Touchdowns": [42.0, 22.0, 45.0, 41.0, 34.0, 26.0, 37.0, 33.0, 35.0],
})

reception_df = pd.DataFrame({
    "Player": [
        "Ceedee Lamb", "Cooper Kupp", "Davante Adams",
        "Justin Jefferson", "Keenan Allen", "Michael Pittman",
        "Stefon Diggs", "Travis Kelce", "Tyreek Hill", "Amon-ra St.Brown"
    ],
    "Total Receptions": [260, 312, 338, 324, 272, 227, 338, 307, 317, 196],
})

yards_df = pd.DataFrame({
    "Player": [
        "Aaron Jones", "Austin Ekeler", "Christian McCaffrey",
        "Davante Adams", "Derrick Henry", "Jonathan Taylor",
        "Justin Jefferson", "Nick Chubb", "Tyreek Hill"
    ],
    "Total Yardage": [5056.0, 5170.0, 5055.0, 5591.0, 6553.0, 5537.0, 5943.0, 4605.0, 6290.0],
})



# Filter the DataFrame for the 2023 season
players_2023_df = week_df[week_df['season'] == 2023]

# Calculate the total touchdowns, receptions, and yardage for each player
# Use .loc to explicitly modify columns in the DataFrame
players_2023_df.loc[:, 'total_touchdowns'] = players_2023_df['reception_td'] + players_2023_df['run_td']
players_2023_df.loc[:, 'total_yards'] = players_2023_df['receiving_yards'] + players_2023_df['rushing_yards']

# Group by player and calculate the averages
average_stats_2023 = (
    players_2023_df.groupby('player_name')
    .agg({
        'total_touchdowns': 'mean',  # Average weekly total touchdowns
        'receptions': 'mean',        # Average weekly receptions
        'total_yards': 'mean'        # Average weekly total yards
    })
    .rename(columns={
        'total_touchdowns': 'Avg Total Touchdowns',
        'receptions': 'Avg Receptions',
        'total_yards': 'Avg Total Yards'
    })
)

# Find the top 3 players with the highest averages for each metric
top_3_touchdowns = average_stats_2023['Avg Total Touchdowns'].nlargest(3)
top_3_receptions = average_stats_2023['Avg Receptions'].nlargest(3)
top_3_yards = average_stats_2023['Avg Total Yards'].nlargest(3)
top_3_fig = make_subplots(
    rows=1, cols=3, 
    subplot_titles=[
        'Most Average Touchdowns', 
        'Most Average Receptions', 
        'Most Average Yardage'
    ],
    column_widths=[0.3, 0.3, 0.3]
)

# Add Top 3 Touchdowns
top_3_fig.add_trace(
    go.Bar(
        x=top_3_touchdowns.index,
        y=top_3_touchdowns.values,
        name='Average Total Touchdowns',
        marker=dict(color=px.colors.sequential.Sunsetdark[0])  # Apply a color
    ),
    row=1, col=1
)

# Add Top 3 Receptions
top_3_fig.add_trace(
    go.Bar(
        x=top_3_receptions.index,
        y=top_3_receptions.values,
        name='Average Receptions',
        marker=dict(color=px.colors.sequential.Sunsetdark[1])  # Apply a color
    ),
    row=1, col=2
)

# Add Top 3 Yardage
top_3_fig.add_trace(
    go.Bar(
        x=top_3_yards.index,
        y=top_3_yards.values,
        name='Average Total Yards',
        marker=dict(color=px.colors.sequential.Sunsetdark[2])  # Apply a color
    ),
    row=1, col=3
)

# Update the layout of the figure
top_3_fig.update_layout(
    title='Top 3 Players Comparison for 2023 Season',
    showlegend=False,
    template='plotly_white',
    height=500,
    bargap=0.3
)

# Step 1: Calculate average stats per season
week_df['Total Touchdowns'] = week_df['reception_td'] + week_df['run_td']
week_df['Total Yards'] = week_df['receiving_yards'] + week_df['rushing_yards']

average_stats = week_df.groupby(['player_name', 'season']).agg(
    Avg_Touchdowns=('Total Touchdowns', 'mean'),
    Avg_Receptions=('receptions', 'mean'),
    Avg_Yards=('Total Yards', 'mean')
).reset_index()

# Step 2: Select three players for comparison
selected_players = ["Justin Jefferson", "Tyreek Hill", "Christian McCaffrey"]
comparison_df = average_stats[average_stats['player_name'].isin(selected_players)]


# Data for the players
players_data = {
    "Player": ["Tyreek Hill", "Davante Adams", "Justin Jefferson"],
    "Seasons Played": [8, 10, 4],
    "Career Receptions (2023)": [772, 898, 392],
    "Career Total TDs (2023)": [85, 96, 30],
    "Career Games (2023)": [136, 155, 60]
}

# Create a DataFrame to hold the players' data
players_df = pd.DataFrame(players_data)

# Calculate per-season averages
players_df["Avg Receptions per Season"] = players_df["Career Receptions (2023)"] / players_df["Seasons Played"]
players_df["Avg TDs per Season"] = players_df["Career Total TDs (2023)"] / players_df["Seasons Played"]
players_df["Avg Games per Season"] = players_df["Career Games (2023)"] / players_df["Seasons Played"]

# Define metrics and titles for visualization
metrics = {
    "Avg Receptions per Season": "Average Receptions per Season",
    "Avg TDs per Season": "Average Touchdowns per Season",
    "Avg Games per Season": "Average Games per Season"
}
colors = px.colors.sequential.Sunsetdark

# Generate individual graphs for each metric
graphs = []
for idx, (metric, title) in enumerate(metrics.items()):
    fig = go.Figure(
        data=[
            go.Bar(
                x=players_df["Player"],
                y=players_df[metric],
                name=title,
                marker_color=colors[idx % len(colors)]  # Cycle through colors
            )
        ],
        layout=go.Layout(
            title=title,
            xaxis_title="Player",
            yaxis_title="Metric Value",
            xaxis=dict(tickangle=0),
            template="plotly_white"
        )
    )
    graphs.append(dcc.Graph(figure=fig)),

# Normalize seasons to start at t=0 for each player
def normalize_seasons(player_data):
    player_data = player_data.sort_values('season')  # Ensure sorted by season
    player_data['t'] = range(len(player_data))  # Create t=0, t=1, etc.
    return player_data

# Apply normalization across groups and avoid future deprecation warnings
normalized_df = year_df.groupby('player_name', group_keys=False).apply(normalize_seasons)

# Calculate total touchdowns and yards for 2023 players
players_2023_df.loc[:, 'total_touchdowns'] = players_2023_df['reception_td'] + players_2023_df['run_td']
players_2023_df.loc[:, 'total_yards'] = players_2023_df['receiving_yards'] + players_2023_df['rushing_yards']


# Function to create the WR baseline comparison graph
def create_wr_baseline_comparison(normalized_df):
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

    # Select players of interest (ensure names are lowercased for matching)
    selected_players = ['justin jefferson', 'davante adams', 'tyreek hill']
    selected_df = normalized_df[normalized_df['player_name'].str.lower().isin(selected_players)]

    # Calculate total performance for selected players
    selected_df['total_touchdowns'] = selected_df['reception_td'] + selected_df['run_td']
    selected_df['total_yards'] = selected_df['receiving_yards'] + selected_df['rushing_yards']
    selected_df['total_performance'] = (
        selected_df['total_touchdowns'] + selected_df['total_yards'] + selected_df['receptions']
    )

    # Merge selected players with the baseline data on 't'
    comparison_df = selected_df.merge(baseline_df, on='t', suffixes=('_player', '_baseline'))

    # Define player colors
    player_colors = {
        'justin jefferson': 'purple',
        'davante adams': 'red',
        'tyreek hill': 'orange',
    }

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
        player_data = comparison_df[comparison_df['player_name'].str.lower() == player_name]
        fig.add_trace(go.Scatter(
            x=player_data['t'],
            y=player_data['total_performance_player'],
            mode='lines+markers',
            name=f"Total Performance: {player_name.title()}",
            line=dict(color=player_colors[player_name]),
            marker=dict(size=8)
        ))

    # Update layout for the figure
    fig.update_layout(
        title='Comparison of Players Against WR Baseline',
        xaxis_title='t (Normalized Time)',
        yaxis_title='Performance Metrics',
        template='plotly_white',
        legend_title='Players',
        hovermode='closest',
        height=600
    )

    return fig

# Call the function to generate the graph
wr_baseline_graph = create_wr_baseline_comparison(normalized_df)


# Step 1: Prepare Data
data = {
    "player_name": ["Justin Jefferson", "Davante Adams", "Tyreek Hill"],
    "age": [24, 31, 29],
    "total_receptions": [453, 898, 766],
    "total_receiving_yards": [6838, 11059, 10710],
    "total_rushing_yards": [31, 0, 813],
    "total_reception_td": [35, 96, 79],
    "total_run_td": [1, 0, 7]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate a composite score as a target variable (proxy for sponsorship suitability)
df["suitability_score"] = (
    df["total_receptions"] * 0.2 + 
    df["total_receiving_yards"] * 0.2 + 
    df["total_rushing_yards"] * 0.2 + 
    df["total_reception_td"] * 0.2 + 
    df["total_run_td"] * 0.2
)

# Features and target variable
X = df[["age", "total_receptions", "total_receiving_yards", "total_rushing_yards", "total_reception_td", "total_run_td"]]
y = df["suitability_score"]

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rf_model = RandomForestRegressor(random_state=42, n_estimators=100)

# Train Random Forest Model
rf_model.fit(X, y)

# Predict for all players
df["rf_predictions"] = rf_model.predict(X)

import plotly.graph_objects as go

# Sample data
players = ["Justin Jefferson", "Davante Adams", "Tyreek Hill"]
rf_predictions = [85.3, 90.7, 78.2]
seasons_played = [4, 10, 8]

# Calculate weighted scores
max_seasons = max(seasons_played)
weights = [max_seasons / sp for sp in seasons_played]
weighted_rf_predictions = [rf * weight for rf, weight in zip(rf_predictions, weights)]



# Step 5: Create the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "NFL Player Analysis"

# Sample Functions for Graphs
def generate_interactive_graph(data):
    fig = px.bar(
        x=list(data.keys()),
        y=list(data.values()),
        title="Total Touchdowns",
        labels={"x": "Player", "y": "Total Touchdowns"},
        template="plotly_white"
    )
    return fig

def generate_receptions_graph(data):
    fig = px.bar(
        x=list(data.keys()),
        y=list(data.values()),
        title="Total Receptions",
        labels={"x": "Player", "y": "Total Receptions"},
        template="plotly_white"
    )
    return fig

def generate_yardage_graph(data):
    fig = px.bar(
        x=list(data.keys()),
        y=list(data.values()),
        title="Total Yardage",
        labels={"x": "Player", "y": "Total Yardage"},
        template="plotly_white"
    )
    return fig


num_touchdown_players = len(touchdown_df)
num_reception_players = len(reception_df)
num_yardage_players = len(yards_df)

# Generate colors for each dataset
touchdown_colors = sample_colorscale('Pinkyl', [i / (num_touchdown_players - 1) for i in range(num_touchdown_players)])
reception_colors = sample_colorscale('Pinkyl', [i / (num_reception_players - 1) for i in range(num_reception_players)])
yardage_colors = sample_colorscale('Pinkyl', [i / (num_yardage_players - 1) for i in range(num_yardage_players)])

num_players = 3  # Number of players
pinkyl_colors = sample_colorscale('Pinkyl', [i / (num_players - 1) for i in range(num_players)])


# Combine all sections into a single layout
prediction_layout = html.Div([
    # Prediction Dashboard Section
    html.Div([
        html.H1("Prediction Dashboard", style={"textAlign": "center", "marginBottom": "30px"}),
        html.H3("2020-2023 Yardage, Touchdown, and Reception Analysis", style={"textAlign": "center"}),
        html.P("Average per Metric", style={"textAlign": "center", "fontSize": "16px"}),
     

         # Add the first graph
    html.Div(
    dcc.Graph(
        id="first-graph",
        figure=create_first_graph(),
        style={"marginTop": "20px"}
    ),
    style={
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center"
    }
),

   html.H3("Top 10 Players per Metric", style={"textAlign": "center", "fontSize": "2px"}),


        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=touchdown_df["Player"],
                        y=touchdown_df["Total Touchdowns"],
                        marker=dict(color=touchdown_colors),  # Use custom colors
                        name="Total Touchdowns"
                    )
                ],
                layout=go.Layout(
                    title="Total Touchdowns",
                    xaxis=dict(title="Player"),
                    yaxis=dict(title="Total Touchdowns"),
                    template="plotly_white"
                )
            )
        )
    ], style={"padding": "20px"}),

    # Receptions Graph Section
    html.Div([
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=reception_df["Player"],
                        y=reception_df["Total Receptions"],
                        marker=dict(color=reception_colors),  # Use custom colors
                        name="Total Receptions"
                    )
                ],
                layout=go.Layout(
                    title="Total Receptions",
                    xaxis=dict(title="Player"),
                    yaxis=dict(title="Total Receptions"),
                    template="plotly_white"
                )
            )
        )
    ], style={"padding": "20px"}),

    # Yardage Graph Section
    html.Div([
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=yards_df["Player"],
                        y=yards_df["Total Yardage"],
                        marker=dict(color=yardage_colors),  # Use custom colors
                        name="Total Yardage"
                    )
                ],
                layout=go.Layout(
                    title="Total Yardage",
                    xaxis=dict(title="Player"),
                    yaxis=dict(title="Total Yardage"),
                    template="plotly_white"
                )
            )
        )
    ], style={"padding": "20px"}),


    # Weekly Data Analysis Section
    html.Div([
        html.H1("2023 Weekly Player Analysis", style={"textAlign": "center", "marginBottom": "30px"}),
        html.H3("Average Weekly Performance Metrics", style={"textAlign": "center"}),
        html.P(
            "This section showcases the top 3 players in average touchdowns, receptions,"
            "and yardage for the 2023 season. The latest season plays a crucial role in evaluating football players,"
            "as their careers often evolve rapidly. By focusing on the most recent data, this analysis provides valuable "
            "insights to inform predictions and strategies for the upcoming 2024 season.",
            style={"textAlign": "center", "fontSize": "16px"}
        ),
        html.Div([
        dcc.Graph(figure=top_3_fig)
    ], style={"padding": "20px"}),

    html.P(
        "Justin Jefferson stands out as the only wide receiver (WR) to rank among the top three players in more than one category across the three graphs.",
        style={"textAlign": "center", "fontSize": "16px", "color": "grey"}
    ),

    # Players Excelling in Multiple Metrics Section
html.Div([
    html.H3("Players Excelling in Multiple Metrics", style={"textAlign": "center"}),
    html.P("These players have appeared in multiple top categories over the last 4 seasons", style={"textAlign": "center"}),
    html.Ul([
        html.Li("Justin Jefferson"),
        html.Li("Davante Adams"),
        html.Li("Tyreek Hill"),
    ], style={"textAlign": "center", "fontSize": "16px", "listStyleType": "none", "padding": "0"}),
], style={"padding": "20px"}),


    # Side-by-Side Graphs for Selected Players
    html.Div([
        html.H1("Player Performance Analysis", style={"textAlign": "center", "marginBottom": "30px"}),
        html.H3("Justin Jefferson vs. Davante Adams vs. Tyreek Hill", style={"textAlign": "center"}),
        html.Div([
    html.Div(dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=["Justin Jefferson", "Davante Adams", "Tyreek Hill"],
                    y=[8, 10, 4],
                    marker=dict(color=pinkyl_colors),  # Apply Pinkyl colors
                    name="Avg Receptions"
                )
            ],
            layout=go.Layout(
                title="Average Receptions per Season",
                xaxis=dict(
                    title="Player",
                    tickangle=0  # Move tickangle here
                ),
                yaxis=dict(title="Avg Receptions"),
                template="plotly_white"
            )
        )
    ), style={"width": "40%", "display": "inline-block", "padding": "5px"}),

    html.Div(dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=["Justin Jefferson", "Davante Adams", "Tyreek Hill"],
                    y=[30, 96, 85],
                    marker=dict(color=pinkyl_colors),  # Apply Pinkyl colors
                    name="Avg Touchdowns"
                )
            ],
            layout=go.Layout(
                title="Average Touchdowns per Season",
                xaxis=dict(
                    title="Player",
                    tickangle=0  # Move tickangle here
                ),
                yaxis=dict(title="Avg Touchdowns"),
                template="plotly_white"
            )
        )
    ), style={"width": "40%", "display": "inline-block", "padding": "5px"}),

    html.Div(dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=["Justin Jefferson", "Davante Adams", "Tyreek Hill"],
                    y=[60, 155, 136],
                    marker=dict(color=pinkyl_colors),  # Apply Pinkyl colors
                    name="Avg Games"
                )
            ],
            layout=go.Layout(
                title="Average Games per Season",
                xaxis=dict(
                    title="Player",
                    tickangle=0  # Move tickangle here
                ),
                yaxis=dict(title="Avg Games"),
                template="plotly_white"
            )
        )
    ), style={"width": "40%", "display": "inline-block", "padding": "5px"})
], style={"display": "flex", "justifyContent": "center", "alignItems": "center"}),

# New WR Baseline Comparison Section
    html.Div([
        html.H1("WR Baseline Comparison", style={"textAlign": "center", "marginBottom": "20px"}),
    dcc.Graph(
        id="wr-baseline-comparison",
        figure=wr_baseline_graph,
        style={"padding": "20px"}
    )
    ]),

    ]),
html.H1("Machine Learning Results: Weighted Random Forest Scores", style={"textAlign": "center", "marginTop": "40px"}),

html.Div([
    dcc.Graph(
        id="weighted-scores-graph",
        figure=go.Figure(
            data=[
                go.Bar(
                    x=players,  # Player names
                    y=weighted_rf_predictions,  # Weighted scores from the updated calculation
                    name="Weighted Random Forest Scores",
                    marker_color="green"  # Removed text and textposition
                )
            ],
            layout=go.Layout(
                title="Weighted Sponsorship Scores by Player",
                xaxis_title="Player",
                yaxis_title="Weighted Sponsorship Score",
                template="plotly_white",
                legend_title="Score Type"
            )
        )
    )
], style={"padding": "20px"}),


    # Sponsorship Recommendation Section
    html.Div([
        html.H3("Player Sponsorship Recommendation", style={"textAlign": "center", "marginTop": "35px"}),
        html.H1("Justin Jefferson", style={"textAlign": "center", "color": "purple", "fontSize": "50px"}),
        html.P(
            "Justin Jefferson’s exceptional yardage and receptions performance make him the ideal"
            "candidate to be Nike's standout athlete for 2024.",
            style={"textAlign": "center", "fontSize": "20px"}
        ),
    ], style={"padding": "20px", "backgroundColor": "#f5f5f5", "borderRadius": "10px"})
])
])
app.layout = prediction_layout


# Run the app
if __name__ == "__main__":
   app.run_server(debug=False)

