'''
Code to reconstruct missing data in dataset/match_history.csv using average
score values for each country.
'''
import csv
import numpy as np
import pandas as pd

HOME_FEATURE_INDEXES = [17, 19, 20, 21]
AWAY_FEATURE_INDEXES = [18, 22, 23, 24]
HOME_TEAM_FIFA_RANK = 5
AWAY_TEAM_FIFA_RANK = 6
PLOT_TITLES = ['Score', 'Goalkeeper', 'Defense', 'Offense', 'Midfield']

def interpolote_missing_data(original_data: str) -> list:
    '''Interpolate completely missing features using Pandas .interpolate()'''

    with open(original_data) as csv_in_file:
        csv_reader = csv.reader(csv_in_file)
        next(csv_reader)

        fifa_ranks = {}
        # features = [scores, gk_scores, def_scores, off_scores, mid_scores]
        features = [[], [], [], []]
        PLOT_TITLES = ['Goalkeeper', 'Defense', 'Offense', 'Midfield']

        for row in csv_reader:
            # Extract the 5 types of score from the given csv for
            home_features = [row[i] for i in HOME_FEATURE_INDEXES]
            away_features = [row[i] for i in AWAY_FEATURE_INDEXES]
            fifa_ranks[row[1]] = int(row[HOME_TEAM_FIFA_RANK])
            fifa_ranks[row[2]] = int(row[AWAY_TEAM_FIFA_RANK])

            # Only append the data to the lists is it is fully intact

            for i in range(4):
                home_val = np.nan if home_features[i] == '' else home_features[i]
                away_val = np.nan if away_features[i] == '' else away_features[i]
                features[i].append((int(row[HOME_TEAM_FIFA_RANK]), (home_val)))
                features[i].append((int(row[AWAY_TEAM_FIFA_RANK]), (away_val)))

        interpolated_features = [[], [], [], []]
        for i, score_type in enumerate(PLOT_TITLES):
            sorted_score = [float(s) for (r,s) in sorted(features[i], key= lambda x: x[0])]
            series = pd.Series(sorted_score)
            interpolated_features[i] = list(series.interpolate())


    return interpolated_features

def write_missing_data(original_data: str, new_data: str,
                       countries: dict) -> None:
    '''
    For each row in the input csv, write to the output csv filling in missisng
    values, with that countries average for that feature.
    '''

    with open(original_data) as csv_in, open(new_data, 'w') as csv_out:
        csv_reader = csv.reader(csv_in)
        csv_writer = csv.writer(csv_out)
        csv_writer.writerow(next(csv_reader))
        
        interpolated_features = interpolote_missing_data(original_data)

        for row in csv_reader:
            for i, val in enumerate(row):
                if val == '':
                    if i in HOME_FEATURE_INDEXES:
                        replace_index = HOME_FEATURE_INDEXES.index(i)
                        if countries[row[1]][replace_index] == 0.0:
                            row[i] = round(interpolated_features[replace_index][int(row[HOME_TEAM_FIFA_RANK])], 1)
                        else:
                            row[i] = countries[row[1]][replace_index]

                    elif i in AWAY_FEATURE_INDEXES:
                        replace_index = AWAY_FEATURE_INDEXES.index(i)
                        if countries[row[2]][replace_index] == 0.0:
                            row[i] = round(interpolated_features[replace_index][int(row[AWAY_TEAM_FIFA_RANK])], 1)
                        else:
                            row[i] = countries[row[2]][replace_index]
            csv_writer.writerow(row)


def get_averages(original_data: str) -> dict:
    '''
    Find average values for each 5 features for each country.
    '''

    countries = {}

    with open(original_data) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        count = 0
        for row in csv_reader:
            if '' in row:
                count += 1
            home_country, away_country = row[1], row[2]
            home_features = [row[i] for i in HOME_FEATURE_INDEXES]
            away_features = [row[i] for i in AWAY_FEATURE_INDEXES]

            if home_country not in countries.keys():
                countries[home_country] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            if away_country not in countries.keys():
                countries[away_country] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

            for i, val in enumerate(home_features):
                if val != '':
                    countries[home_country][i][0] += float(val)
                    countries[home_country][i][1] += 1

            for i, val in enumerate(away_features):
                if val != '':
                    countries[away_country][i][0] += float(val)
                    countries[away_country][i][1] += 1

        for country, vals in countries.items():
            averages = [round(val / max(1, count), 1) for [val, count] in vals]
            countries[country] = averages
        for country, vals in countries.items():
            if 0.0 in vals:
                print(f'{country} has a completely missing value.')
    return countries


def reconstruct(original_data: str, new_data: str) -> None:
    '''
    Reconstruct the missing data in original_data and write to new
    new_data csv.
    '''

    countries = get_averages(original_data)
    write_missing_data(original_data, new_data, countries)


if __name__ == '__main__':
    reconstruct('dataset/match_history.csv', 'reconstructed_match_history.csv')
