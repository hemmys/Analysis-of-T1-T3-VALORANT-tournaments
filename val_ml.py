"""
Eli Lingat, Claudine Dulay, Emily Estrada
This file contains functions for RQ3, which deal with machine learning
and their use in predicting player data and outcomes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import random
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


def scoreboard_predict(df):
    """
    Takes in a Scoreboard Dataframe and predicts a sample of top players
    in future years. Returns the resulting DataFrame and the test accuracy.
    """
    player_df = df[['PlayerName', 'ACS', 'Agent', 'ADR',
                    'Econ', 'TeamAbbreviation',
                    'Kills', 'Deaths', 'Assists']]
    player_df = player_df.dropna()
    model = DecisionTreeClassifier()
    features = player_df.loc[:, player_df.columns != 'PlayerName']
    features = pd.get_dummies(features)
    labels = player_df['PlayerName']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.33)
    model.fit(features_train, labels_train)
    test_pred = model.predict(features_test)
    res_df = features_test
    res_df['ML_PlayerName'] = test_pred
    res_df['True_PlayerName'] = labels_test
    res_df = res_df.sort_values(by='ACS', axis=0, ascending=False)
    test_acc = accuracy_score(labels_test, test_pred)
    return res_df, test_acc


def fantasy_team_df_prep(df, stat, players):
    """
    Takes in a Scoreboard DataFrame, a given stat, and a list of 5 players,
    and returns a DataFrame that is prepared for model use.
    """
    df_list = []
    for name in players:
        mask = (df['PlayerName'] == name)
        roster = df[mask]
        roster = roster[['PlayerName', stat]]
        roster.rename(columns={stat: f'{name}_{stat}'}, inplace=True)
        df_list.append(roster)

    team = pd.DataFrame()
    for i in range(len(df_list)):
        if i == 0:
            team = df_list[i]
        else:
            team = team.merge(df_list[i], how='outer')
    team = team.fillna(0)
    team = pd.get_dummies(team)
    team[stat] = 0

    for name in players:
        team[stat] += team[f'{name}_{stat}']

    return team


def fantasy_team(df, stat='ACS',
                 players=['bdog', 's0m', "TenZ", 'Reduxx', 'ChurmZ']):
    """
    Takes in a list of 5 players, a dataframe of Scoreboard data, and a stat,
    and returns the predicted team average of the passed in stat.
    The Scoreboard DataFrame should at least have a column
    named 'PlayerName', and a column labelled by the passed in stat.
    """
    team = fantasy_team_df_prep(df, stat, players)

    model = DecisionTreeRegressor()
    features = team.loc[:, team.columns != stat]
    labels = team[stat]

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)

    model.fit(features_train, labels_train)
    test_pred = model.predict(features_test)

    res_df = features_test
    res_df[f'ML_{stat}'] = test_pred
    res_df[f'True_{stat}'] = labels_test
    test_acc = mean_squared_error(labels_test, test_pred)
    average = np.average(test_pred)

    return average, test_acc


def match_predict(df, teamA, teamB, stats):
    """
    Takes in a Scoreboard DataFrame, 2 teams of 5 players, and a list of
    stats, and prints out the predicted result of a match between the
    two teams based on the averages of each stat passed in, taking into
    account random map choice. The Scoreboard DataFrame must have a
    PlayerName column, a Map column, and columns named after the various
    stats passed in. Also returns a string corresponding to the team that won.
    """
    teamA_score = 0.0
    teamB_score = 0.0
    teamA_wins = 0
    teamB_wins = 0
    rms_A_arr = []
    rms_B_arr = []
    match_end = False
    i = 0

    map_series = df[df['Map'] != 'TBD']
    map_list = map_series['Map'].unique().tolist()
    maps = random.sample(map_list, 3)

    while match_end is False:
        # loop for one game
        for stat in stats:
            tA_stat_avg, rms_A = fantasy_team(df[df['Map'] == maps[i]],
                                              stat, teamA)
            tB_stat_avg, rms_B = fantasy_team(df[df['Map'] == maps[i]],
                                              stat, teamB)
            if tA_stat_avg > tB_stat_avg:
                teamA_score += 1
            elif tA_stat_avg < tB_stat_avg:
                teamB_score += 1
            else:
                teamA_score += .5
                teamB_score += .5
            rms_A_arr.append(rms_A)
            rms_B_arr.append(rms_B)

        if teamA_score > teamB_score:
            print(f'Game {i+1}: Team A won {maps[i]}')
            teamA_wins += 1
        elif teamA_score < teamB_score:
            print(f'Game {i+1}: Team B won {maps[i]}')
            teamB_wins += 1
        else:
            coinflip = np.random.randint(2)
            if coinflip == 0:
                print(f'Game {i+1}: Team A won {maps[i]} (OT)')
                teamA_wins += 1
            else:
                print(f'Game {i+1}: Team B won {maps[i]} (OT)')
                teamB_wins += 1

        i += 1
        teamA_score = 0.0
        teamB_score = 0.0
        if teamA_wins == 2 or teamB_wins == 2:
            match_end = True

    print(f'Team A {teamA_wins} - {teamB_wins} Team B')

    rms_A_arr = np.array(rms_A_arr)
    rms_B_arr = np.array(rms_B_arr)
    avg_rms_A = np.average(rms_A_arr)
    avg_rms_B = np.average(rms_B_arr)

    if teamA_wins > teamB_wins:
        return 'Team A', avg_rms_A, avg_rms_B
    elif teamA_wins < teamB_wins:
        return 'Team B', avg_rms_A, avg_rms_B


def match_trend(df, teamA, teamB, stats, n=50):
    """
    Takes in a Scoreboard DataFrame, 2 teams of 5 players, a list of stats,
    and the number of matches, and creates a bar chart comparing the wins
    between the two teams, and a scatter plot showing the average RMS
    of each game.
    """
    teamA_matches = 0.0
    teamB_matches = 0.0
    average_rms_A = []
    average_rms_B = []
    for i in range(n):
        winner = match_predict(df, teamA, teamB, stats)
        if winner[0] == 'Team A':
            teamA_matches += 1
        else:
            teamB_matches += 1
        average_rms_A.append(winner[1])
        average_rms_B.append(winner[2])

    labels = ['Team A', 'Team B']
    x = np.arange(len(labels))
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(11, 6)
    ax[0].bar(0, teamA_matches, label='Team A')
    ax[0].bar(1, teamB_matches, label='Team B')
    ax[0].set_xlabel('Teams')
    ax[0].set_xticks(x, labels=labels)
    ax[0].set_ylabel('Wins')
    ax[0].set_title('Predicted Match Wins Between Team A and Team B')
    ax[0].legend()

    ax[1].scatter(range(1, len(average_rms_A)+1),
                  average_rms_A, label='Team A RMS')
    ax[1].scatter(range(1, len(average_rms_B)+1),
                  average_rms_B, label='Team B RMS')
    ax[1].set_xlabel('Match Number')
    ax[1].set_ylabel('Average RMS')
    ax[1].set_title('Average RMS of Team ML Models Across all Given Stats')
    ax[1].legend()

    plt.show()
    plt.savefig('match_predict_bar_chart.png', bbox_inches='tight')


def main():
    scoreboard_df, rounds_df, games_df, matches_df, merge_df = df_creator()
    scoreboard_predict(scoreboard_df)
    fantasy_team(merge_df)
    team1 = ['bdog', 's0m', "TenZ", 'Reduxx', 'ChurmZ']
    team2 = ['NamGoku', 'dino', 'nAts', 'Lin', 'Jerk']
    match_predict(merge_df, team1, team2, ['ACS', 'Econ', 'Kills', 'ADR'])
    match_trend(merge_df, team1, team2, ['ACS', 'Kills', 'Econ', 'ADR'])


if __name__ == '__main__':
    main()
