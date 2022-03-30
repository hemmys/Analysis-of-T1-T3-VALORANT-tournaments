"""
Eli Lingat, Claudine Dulay, Emily Estrada
This function contains a function that creates pandas dataframes
for use throughout the rest of the project.
"""
import pandas as pd
import sqlite3 as sql


def val_df_gen(db_name):
    """
    Takes in pro player data from the website:
    https://www.kaggle.com/visualize25/valorant-pro-matches-full-data
    and returns dataframes in the db file from the link, as well
    as a dataframe that consists of all data from the dataframe.
    """

    val_sql = sql.connect('valorant.sqlite')

    scoreboard_df = pd.read_sql("""
    SELECT
        *
    FROM
        Game_Scoreboard
    """, val_sql)
    rounds_df = pd.read_sql("""
        SELECT
            *
        FROM
            Game_Rounds
    """, val_sql)
    games_df = pd.read_sql("""
        SELECT
            *
        FROM
            Games
    """, val_sql)
    matches_df = pd.read_sql("""
        SELECT
            *
        FROM
            Matches
    """, val_sql)

    val_sql.close()

    merge_df = matches_df.merge(games_df)
    merge_df = merge_df.merge(rounds_df)
    merge_df = merge_df.merge(scoreboard_df)
    merge_df['MatchID'] = merge_df['MatchID'].astype(int)

    scoreboard_df = scoreboard_df.dropna()
    rounds_df = rounds_df.dropna()
    games_df = games_df.dropna()
    matches_df = matches_df.dropna()
    merge_df = merge_df.dropna()

    scoreboard_df.to_csv('sb_df.csv')
    rounds_df.to_csv('r_df.csv')
    games_df.to_csv('g_df.csv')
    matches_df.to_csv('ma_df.csv')
    merge_df.to_csv('me_df.csv')

    return scoreboard_df, rounds_df, games_df, matches_df, merge_df
