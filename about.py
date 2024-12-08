from dash import html

about_layout = html.Div(
    [
        html.H1("About Stats to Stars", style={"textAlign": "center", "marginBottom": "20px"}),
        html.Div(
            [
                html.H3("Welcome to Stats to Stars"),
                html.P(
                    "Stats to Stars is an interactive platform that provides in-depth insights and "
                    "visualizations of NFL player performance. Designed for fans, analysts, coaches, and "
                    "sponsors, the app allows users to explore, analyze, and compare the performance of top "
                    "players across recent seasons."
                ),
                html.H3("Purpose"),
                html.P(
                    "The app aims to bridge the gap between raw data and actionable insights by offering visually "
                    "compelling dashboards and analytics. It empowers users to:"
                ),
                html.Ul(
                    [
                        html.Li("Analyze player performance trends across seasons."),
                        html.Li("Identify top players based on key metrics like touchdowns, receptions, and yardage."),
                        html.Li("Compare players' career and weekly performance to make informed decisions."),
                        html.Li("Explore interactive charts to uncover hidden patterns and insights."),
                    ]
                ),
                html.H3("Key Features"),
                html.Ul(
                    [
                        html.Li("Seasonal Analysis: Explore the top performers for each season (2012-2023) in categories like touchdowns, receptions, and total yardage."),
                        html.Li("Weekly Performance: Dive into detailed weekly stats for the 2023 season, including averages and top 3 players for touchdowns, receptions, and yardage."),
                        html.Li("Career Comparisons: Compare career stats of standout players, including their average touchdowns, receptions, and games per season."),
                        html.Li("Interactive Graphs: Intuitive and interactive visualizations for seamless exploration of player data."),
                        html.Li("Consistent Performers: Identify players who consistently rank in the top 10 across multiple seasons."),
                    ]
                ),
                html.H3("Who Is This For?"),
                html.Ul(
                    [
                        html.Li("Fans: Discover how your favorite players stack up against the competition."),
                        html.Li("Coaches and Analysts: Use insights to strategize and refine gameplay."),
                        html.Li("Sponsors: Find the best players to endorse by evaluating their consistent performance and marketability."),
                        html.Li("Fantasy Football Enthusiasts: Make informed picks and trades for your fantasy team."),
                    ]
                ),
                html.H3("Data Sources"),
                html.P(
                    "All data used in this app is sourced from reliable platforms, including the NFL and Kaggle repositories, ensuring accuracy and up-to-date information."
                ),
                html.H3("Acknowledgments"),
                html.P(
                    "This app was developed as part of an educational and exploratory project, with contributions from "
                    "Salomé Rivas and inspired by the passion for sports analytics and data visualization."
                ),
            ],
            style={"lineHeight": "1.6", "fontSize": "16px", "margin": "0 50px"},
        ),
    ],
    style={"padding": "20px", "backgroundColor": "#f9f9f9"},
)
