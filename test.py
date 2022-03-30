"""
Eli Lingat, Claudine Dulay, Emily Estrada
This file contains test functions for RQ1, RQ2, and RQ3.
We will test if the graphs match the expected graphs, and
getting the dataframes together.
"""
import pandas as pd
import graphs
from cse163_utils import assert_equals
import val_ml
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


def test_graphs(df):
    '''
    This function tests each of our data visualizations (graphs, plots, etc.)
    and if they are outputted as our expected graphs.
    '''
    graphs.acs_per_map(df)
    graphs.acs_per_agent(df)
    graphs.total_map_picks(df)
    graphs.agent_picks_per_map(df)
    graphs.total_agent_picks(df)
    print('Data Visuals for Research Questions Finished Graphing!')


def test_fantasy_team_df_prep(df):
    '''
    This function tests the dataframe functions used for machine learning
    and if they are identical to the expected output.
    '''
    expected = pd.read_csv('ft_df_expected_output.csv')
    assert_equals(len(expected), len(val_ml.fantasy_team_df_prep(df)))


def main():
    scoreboard_df, rounds_df, games_df, matches_df, merge_df = df_creator()
    df = merge_df
    test_graphs(df)
    test_fantasy_team_df_prep(df)


if __name__ == '__main__':
    main()
