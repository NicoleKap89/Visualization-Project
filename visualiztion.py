import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide")
# Title
st.title("World Cup Comparisons Graph Presentation")
st.markdown("""
The World Cup is one of the most prestigious and widely viewed sporting events globally, bringing together teams from various countries to compete at the highest level of international football. Performance metrics such as goals, assists, and defensive actions are crucial in evaluating both individual and team performances. Understanding these metrics can provide valuable insights into team consistency, team strategies, player contributions, and overall team effectiveness across different tournaments.
""")

@st.cache_data
def load_data():
    data = pd.read_csv('world_cup_comparisons.csv')

    # List of performance metrics
    metrics_columns = [
        'goals_z', 'xg_z', 'crosses_z', 'boxtouches_z', 'passes_z', 'progpasses_z',
        'takeons_z', 'progruns_z', 'tackles_z', 'interceptions_z', 'clearances_z',
        'blocks_z', 'aerials_z', 'fouls_z', 'fouled_z', 'nsxg_z'
    ]

    # Convert negative values to positive
    for column in metrics_columns:
        min_value = data[column].min()
        if (min_value < 0):
            data[column] += abs(min_value)

    # Calculate the performance metric as the sum of the specified metrics
    data['performance'] = data[metrics_columns].sum(axis=1)
    data['attack_metrics'] = data[['goals_z', 'xg_z', 'crosses_z', 'boxtouches_z']].sum(axis=1)
    data['defense_metrics'] = data[['tackles_z', 'interceptions_z', 'clearances_z', 'blocks_z', 'aerials_z']].sum(axis=1)
    data['passing_metrics'] = data[['passes_z', 'progpasses_z', 'takeons_z', 'progruns_z']].sum(axis=1)

    return data


data = load_data()

# Filter Data
def filtered_data(data, above_x_teams=0, above_y_players=0):
    """
    Filter the data based on the number of seasons played by teams and players.

    Parameters:
    data (DataFrame): The dataset containing team and player data.
    above_x_teams (int): The minimum number of seasons a team must have played.
    above_y_players (int): The minimum number of seasons a player must have played.

    Returns:
    DataFrame: The filtered dataset.
    """

    # Filter teams
    if above_x_teams > 0:
        teams_in_multiple_seasons = data.groupby('team')['season'].nunique()
        teams_in_multiple_seasons = teams_in_multiple_seasons[teams_in_multiple_seasons > above_x_teams].index
        data = data[data['team'].isin(teams_in_multiple_seasons)]

    # Filter players
    if above_y_players > 0:
        players_in_multiple_seasons = data.groupby('player')['season'].nunique()
        players_in_multiple_seasons = players_in_multiple_seasons[players_in_multiple_seasons > above_y_players].index
        data = data[data['player'].isin(players_in_multiple_seasons)]

    return data


# Graph 1:
st.header("Trend of Attack, Defense, and Passing Metrics for Teams Across World Cup Seasons")
# Filter teams that participated in 8 or more seasons
filtered_data_teams_1 = filtered_data(data, above_x_teams=8)

# Group by team and season, and calculate the mean of attack_metrics, defense_metrics, and passing_metrics
team_metrics_trend = filtered_data_teams_1.groupby(['team', 'season'])[
    ['attack_metrics', 'defense_metrics', 'passing_metrics']].mean().reset_index()

# Set up the color palette and style
sns.set_palette("husl")
sns.set_style("whitegrid")

# Define colors for attack, defense, and passing metrics
colors = {'attack_metrics': 'blue', 'defense_metrics': 'green', 'passing_metrics': 'red'}

# Create widgets for filtering
teams = sorted(team_metrics_trend['team'].unique())
teams_with_all_option = ['All Teams'] + teams  # Add 'All Teams' option
metrics = ['attack_metrics', 'defense_metrics', 'passing_metrics']

# Streamlit selectors
selected_teams = st.multiselect('Select Teams', options=teams_with_all_option, default='All Teams')
selected_metrics = st.multiselect('Select Metrics', options=metrics, default=metrics)


def update_plot(selected_teams, selected_metrics):
    # Handle 'All Teams' option
    if 'All Teams' in selected_teams:
        selected_teams = teams  # Select all teams

    # Filter data based on selected teams and metrics
    filtered_trend = team_metrics_trend[team_metrics_trend['team'].isin(selected_teams)]

    # Plot using Seaborn FacetGrid
    g = sns.FacetGrid(filtered_trend, col='team', col_wrap=4, height=4, aspect=1.5, sharey=False)

    if 'attack_metrics' in selected_metrics:
        g.map_dataframe(sns.lineplot, x='season', y='attack_metrics', marker='o', color=colors['attack_metrics'],
                        label='Attack Metrics')
    if 'defense_metrics' in selected_metrics:
        g.map_dataframe(sns.lineplot, x='season', y='defense_metrics', marker='s', color=colors['defense_metrics'],
                        label='Defense Metrics')
    if 'passing_metrics' in selected_metrics:
        g.map_dataframe(sns.lineplot, x='season', y='passing_metrics', marker='^', color=colors['passing_metrics'],
                        label='Passing Metrics')

    # Add title and labels with increased font sizes
    g.fig.suptitle('Trend of Attack, Defense, and Passing Metrics for Teams Across World Cup Seasons', fontsize=28)
    g.set_axis_labels('World Cup Season (Year)', 'Metrics', fontsize=18)

    # Adjust subplot titles
    for ax in g.axes.flat:
        ax.set_title(ax.get_title().split('=')[1], fontsize=14)

    # Add legend to the top right
    handles, labels = g.axes[0].get_legend_handles_labels()
    g.fig.legend(handles, labels, loc='upper right', title='Metrics', fontsize=16)

    # Adjust font sizes of team names in the facet grid
    for ax in g.axes.flat:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(g.fig)

# Update plot based on user selection
update_plot(selected_teams, selected_metrics)

st.header("Mean Goals Metric for Teams Across World Cup Seasons")
filtered_data = filtered_data(data, above_x_teams=8)
def graph2():
    # Find teams that appear in more than 5 seasons

    # Define bins (intervals for seasons)
    bins = [1966, 1970, 1980, 1990, 2000, 2010, 2020]  # Adjusted bins to start from 1966
    bin_labels = [f'{start}-{end-1}' for start, end in zip(bins[:-1], bins[1:])]

    # Bin the seasons
    filtered_data['season_bin'] = pd.cut(filtered_data['season'], bins=bins, labels=bin_labels, right=False)

    # Group by team and bin, then calculate the mean goals_z for each group
    team_xg_mean_bins = filtered_data.groupby(['team', 'season_bin'])['goals_z'].mean().reset_index()

    # Streamlit selectors
    selected_teams = st.multiselect('Select Teams', options=filtered_data['team'].unique(), default=filtered_data['team'].unique())

    # Filter data based on selected teams
    filtered_team_data = team_xg_mean_bins[team_xg_mean_bins['team'].isin(selected_teams)]

    # Plot the line plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use seaborn lineplot for the visualization
    sns.lineplot(x='season_bin', y='goals_z', hue='team', data=filtered_team_data, marker='o', ax=ax)

    # Add title and labels
    ax.set_title('Change in Mean Goals Metric for Teams Appearing in More Than 5 World Cup Seasons (Binned)')
    ax.set_xlabel('Season Bin')
    ax.set_ylabel('Mean Goals_z')

    # Show legend outside the plot
    ax.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show plot
    st.pyplot(fig)

graph2()



st.header("Preformance HeatMap for Players across the Years")
# Dropdown options for teams
team_options = data['team'].unique()
# Streamlit selectors
selected_team = st.selectbox('Select Team', options=team_options)

# Function to update the heatmap based on dropdown selection
# Function to update the heatmap based on dropdown selection
def update_heatmap(selected_team):
    # Filter data for the selected team
    team_data = data[data['team'] == selected_team]

    # Count number of World Cups played by each player
    player_world_cups = team_data.groupby('player')['season'].nunique()

    # Filter players who played in 3 or more World Cups
    players_with_three_plus_world_cups = player_world_cups[player_world_cups >= 3].index

    # Filter team data for selected players
    filtered_team_data = team_data[team_data['player'].isin(players_with_three_plus_world_cups)]

    # Pivot the data for the heatmap
    pivot_data = filtered_team_data.pivot_table(index='player', columns='season', values='performance', fill_value=0)

    # Create the heatmap figure
    fig = px.imshow(pivot_data, labels=dict(x="Season", y="Player", color="Performance"),
                    title=f"Player Performance for {selected_team} Over the Years",
                    color_continuous_scale='dense',
                    height=600,  # Adjust height as needed
                    width=1000)  # Adjust width as needed

    return fig


# Update plot based on user selection
heatmap_figure = update_heatmap(selected_team)
st.plotly_chart(heatmap_figure, use_container_width=True)






