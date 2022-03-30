# Analysis-of-T1-T3-VALORANT-tournaments

## Overview
Our project explores the use of graphs, machine learning, and tables in order to view 
statistics about T1-T3 VALORANT tournaments. In this file, `README.md`, we will explain 
the use how to use the program and achieve the results we reviewed.

## Instructions 
In order to achieve the results our project viewed,
the program should:

* Make sure the install all libraries needed which are numpy, pandas, matplotlib.pyplot, seaborn,
and sqlite3. 
* Install data through reading the sqlite3 file to db file.
* Make sure to create the df files by taking them from their cvs files 
and create them into db files.
* Merge all columns that would be needed for viewing the data such as
Agent, Map, ACS, and more. 
* Group by Map and ACS to get the average for each
* Create and save an sns.relplot for the ACS per Map
* Group by Agent and ACS to get the average for each
* Create and save an sns.stripplot for the ACS per Agent
* Create and save an sns.catplot for Map and Total Map Picks
* Create and save an sns.catplot for Agent and Total Agent Picks
* Create and save an sns.catplot for Agent and Total Agent Picks for overall
* For player predictions, install libraries from sklearn.tree and import DecisionTreeClassifier, 
DecisionTreeRegressor, from sklearn.model_selection and import train_test_split,
from sklearn.metrics and import mean_squared_error, accuracy_score, and import random. 
* Use the Scoreboard Dataframe and predicts a sample of top players through looking at values 
such as PlayerName, ACS, Agent, ADR, Econ, TeamAbbreviation, Kills, Deaths, and Assists.
* Create a roster using Scoreboard Dataframe by player name and a different value from above. 
* Create a DecisionTreeRegressor model using five players and value in order to see the prediction.
* Insert five player names and a value such as ACS in order to create a 'fantasy team' and predicted statistics. 
* Inserting five player names for two teams and values to see which team would win each map and with their averages. 
* Creates a bar chart comparing the wins between two teams and each match.
