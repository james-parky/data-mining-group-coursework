'''
Plot 5 graphs of score, goalkeeper score, offense score, defense score and
midfield score against FIFA rank for each country that appears in the input
.csv file.
'''

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


HOME_FEATURE_INDEXES = [9, 17, 19, 20, 21]
AWAY_FEATURE_INDEXES = [10, 18, 22, 23, 24]
HOME_TEAM_FIFA_RANK = 5
AWAY_TEAM_FIFA_RANK = 6
PLOT_TITLES = ['Score', 'Goalkeeper', 'Defense', 'Offense', 'Midfield']


def plot_fifa_score(original_data: str) -> None:
    '''
    Pull data from the given csv and plot 5 graphs, of score, offense score,
    defense score, midfield score and goalkeeper score against FIFA rank.
    '''
    with open(original_data) as csv_in_file:
        csv_reader = csv.reader(csv_in_file)
        # Skip the column names when reading
        next(csv_reader)

        ranks = []
        fifa_ranks = {}
        # features = [scores, gk_scores, def_scores, off_scores, mid_scores]
        features = [[], [], [], [], []]
        plot_colours = ['b', 'r', 'g', 'y', 'm']
        PLOT_TITLES = ['Score', 'Goalkeeper', 'Defense', 'Offense', 'Midfield']

        for row in csv_reader:
            # Extract the 5 types of score from the given csv for
            home_features = [row[i] for i in HOME_FEATURE_INDEXES]
            away_features = [row[i] for i in AWAY_FEATURE_INDEXES]
            fifa_ranks[row[1]] = int(row[HOME_TEAM_FIFA_RANK])
            fifa_ranks[row[2]] = int(row[AWAY_TEAM_FIFA_RANK])

            # Only append the data to the lists is it is fully intact
            if '' not in home_features and '' not in away_features:
                ranks.append(int(row[HOME_TEAM_FIFA_RANK]))
                ranks.append(int(row[AWAY_TEAM_FIFA_RANK]))
                for i in range(5):
                    features[i].append(float(home_features[i]))
                    features[i].append(float(away_features[i]))

        # Create a figure with 5 subplots - one for each type of score
        fig, axs = plt.subplots(5, 1, constrained_layout=True)
        fig.suptitle('A plot of different score metrics by FIFA ranking')

        # Zip the data, colour and title lists for the subplots
        plot_data = zip(plot_colours, PLOT_TITLES, features)
        for i, data in enumerate(plot_data):
            # For each score, plot it against the corresponding FIFA ranking
            axs[i].scatter(ranks, data[2], s=10, c=data[0], marker='.')
            axs[i].set_title(f'{data[1]} by FIFA rank')
            axs[i].set_xlabel('FIFA rank')
            axs[i].set_ylabel('Score')

        plt.show()

if __name__ == '__main__':
    plot_fifa_score('dataset/match_history.csv')
