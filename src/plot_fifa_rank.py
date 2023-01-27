'''
Plot 5 graphs of score, goalkeeper score, offense score, defense score and
midfield score against FIFA rank for each country that appears in the input
.csv file.
'''

import csv
import matplotlib.pyplot as plt


HOME_FEATURE_INDEXES = [17, 19, 20, 21]
AWAY_FEATURE_INDEXES = [18, 22, 23, 24]
HOME_TEAM_FIFA_RANK = 5
AWAY_TEAM_FIFA_RANK = 6
PLOT_TITLES = ['Goalkeeper', 'Defense', 'Offense', 'Midfield']
PLOT_COLOURS = 'rbgm'


def plot_fifa_score(original_data: str) -> None:
    '''
    Pull data from the given csv and plot 5 graphs, of score, offense score,
    defense score, midfield score and goalkeeper score against FIFA rank.
    '''
    with open(original_data) as csv_in_file:
        csv_reader = csv.reader(csv_in_file)
        next(csv_reader)

        ranks = []
        features = {'Goalkeeper': [], 'Defense': [], 'Offense': [], 'Midfield': []}

        for row in csv_reader:
            home_features = [row[i] for i in HOME_FEATURE_INDEXES]
            away_features = [row[i] for i in AWAY_FEATURE_INDEXES]

            if '' not in home_features and '' not in away_features:
                ranks.append(int(row[HOME_TEAM_FIFA_RANK]))
                ranks.append(int(row[AWAY_TEAM_FIFA_RANK]))
                for i, score in enumerate(PLOT_TITLES):
                    new_points = [float(home_features[i]), float(away_features[i])]
                    features[score].extend(new_points)

        fig, axs = plt.subplots(4, 1, constrained_layout=True)
        fig.suptitle('A plot of different score metrics by FIFA ranking')

        plot_data = zip(PLOT_COLOURS, features.items())
        for i, (colour, (title, data)) in enumerate(plot_data):
            axs[i].scatter(ranks, data, s=10, c=colour, marker='.')
            axs[i].set_title(f'{title} by FIFA rank')
            axs[i].set_xlabel('FIFA rank')
            axs[i].set_ylabel('Score')

        plt.show()


if __name__ == '__main__':
    plot_fifa_score('dataset/match_history.csv')
