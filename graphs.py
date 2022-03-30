"""
Eli Lingat, Claudine Dulay, Emily Estrada
This file contains functions for RQ1 and RQ2,
which deal with creating graphs
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from df_gen import val_df_gen


def df_creator():
    """
    Attempts to read .csv files for the dataframes. If there are none,
    attempts to generate them by reading them in from the sql db file.
    """
    try:
        print('Reading .csv files')
        sb_df = pd.read_csv('sb_df.csv')
        r_df = pd.read_csv('r_df.csv')
        g_df = pd.read_csv('g_df.csv')
        ma_df = pd.read_csv('ma_df.csv')
        me_df = pd.read_csv('me_df.csv')
    except IOError:
        print('.csv files not found. Creating .csv files')
        sb_df, r_df, g_df, ma_df, me_df = val_df_gen('valorant.sqlite')

    return sb_df, r_df, g_df, ma_df, me_df


# functions for RQ1
def acs_per_map(df):
    '''
    This function averages the ASC for each map and creates a graph for
    each ACS per map in Valorant. The graph looks at the ACS for each map
    and its differences with x-axis and y-axis labels.
    '''
    # ACS per map numerical value (RQ1)
    df.groupby('Map')['ACS'].mean()

    # graph for ACS per map (RQ1)
    sns.relplot(data=df, x="ACS", y="Map", hue="Map")

    plt.title("ACS per Map")
    plt.xlabel("ACS")
    plt.ylabel("Map")

    plt.savefig('ACS_per_map.png')


def acs_per_agent(df):
    '''
    This function makes looks at the mean of agents and ACS
    while also creating a graph for the ACS per agent in Valorant.
    The graph compares the ACS to each agent on average with x-axis and
    y-axis labels.
    '''
    # ACS per agent numerical value (RQ1)
    df.groupby('Agent')['ACS'].mean()

    # graph for ACS per agent (RQ1)
    sns.stripplot(data=df, x="ACS", y="Agent", hue="Agent")

    plt.title("ACS per Agent")
    plt.xlabel("ACS")
    plt.ylabel("Agent")
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)

    plt.savefig('ACS_per_agent.png')


# functions for RQ2
def total_map_picks(df):
    '''
    This function makes a graph for each of the agents in Valorant and graphs
    the total number of times the agent was picked in a certain map (x-axis).
    Each map is listed on the x-axis.
    '''
    g = sns.catplot(x="Map", col="Agent", col_wrap=4,
                    data=df, kind="count", hue='Map', dodge=False)
    g.set_axis_labels("Map", "Total Map Picks")
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0)
    g.savefig('total_map_picks.png')
    plt.show()


def agent_picks_per_map(df):
    '''
    This function makes a graph for each of the maps in Valorant and graphs
    the total number of times each agent was picked (y-axis) in a certain map.
    Each agent is listed on the x-axis.
    '''
    g = sns.catplot(x="Agent", col="Map", col_wrap=4,
                    data=df, kind="count", hue='Agent', dodge=False)
    g.set_axis_labels("Agent", "Total Agent Picks")
    g.set_xticklabels(rotation=65, horizontalalignment='right')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0)
    g.savefig('agent_picks_per_map.png')
    plt.show()


def total_agent_picks(df):
    '''
    This function makes a graph for each agent (x-axis) and
    their overall total picks (y-axis) in comparison of each other
    so we can give insight on which agents are being picked the most.
    '''
    g = sns.catplot(x="Agent", data=df, kind="count", hue='Agent', dodge=False)
    g.set_axis_labels("Agent", "Total Agent Picks")
    g.set_xticklabels(rotation=65, horizontalalignment='right')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left', borderaxespad=0)
    g.savefig('total_agent_picks.png')
    plt.show()


def main():
    scoreboard_df, rounds_df, games_df, matches_df, merge_df = df_creator()
    df = merge_df[['EventName', 'PlayerName', 'Agent', 'Map', 'Winner',
                   'Team1', 'Team2', 'Team1_MapScore', 'Team2_MapScore',
                   'ACS', 'Kills', 'Deaths', 'Assists', 'Econ',
                   'Team1_TotalRounds', 'Team2_TotalRounds', 'MatchID',
                   'GameID', 'Team1ID', 'Team2ID', 'HS_Percent',
                   'Num_5Ks', 'ADR', 'Plants', 'Defuses',
                   'Team1_RoundsOT', 'Team2_RoundsOT']]
    df = df.dropna()
    acs_per_map(df)
    acs_per_agent(df)
    total_map_picks(df)
    agent_picks_per_map(df)
    total_agent_picks(df)


if __name__ == '__main__':
    main()
