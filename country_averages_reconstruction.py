'''
Code to reconstruct missing data in dataset/match_history.csv using average
score values for each country
'''

import csv

HOME_FEATURE_INDEXES = [9, 17, 19, 20, 21]
AWAY_FEATURE_INDEXES = [10, 18, 22, 23, 24]


def write_missing_data(original_data: str, reconstructed_data: str,
                        countries: dict) -> None:
    '''
    For each missing data, write the corresponding countries average
    '''

    with open(original_data) as csv_in_file, open(reconstructed_data, 'w') as csv_out_file:
        csv_reader = csv.reader(csv_in_file)
        csv_writer = csv.writer(csv_out_file)
        csv_writer.writerow(next(csv_reader))

        for row in csv_reader:
            for i, val in enumerate(row):
                if val == '':
                    if i in HOME_FEATURE_INDEXES:
                        replace_index = HOME_FEATURE_INDEXES.index(i)
                        row[i] = countries[row[1]][replace_index]
                    elif i in AWAY_FEATURE_INDEXES:
                        replace_index = AWAY_FEATURE_INDEXES.index(i)
                        row[i] = countries[row[2]][replace_index]
            csv_writer.writerow(row)


def get_averages(original_data: str) -> dict:
    '''
    Find average values for each 5 features for each country
    '''

    countries = {}

    with open(original_data) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            home_country, away_country = row[1], row[2]
            home_features = [row[i] for i in HOME_FEATURE_INDEXES]
            away_features = [row[i] for i in AWAY_FEATURE_INDEXES]

            if home_country not in countries.keys():
                countries[home_country] = [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]]
            if away_country not in countries.keys():
                countries[away_country] = [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]]

            for i, val in enumerate(home_features):
                if val != '':
                    countries[home_country][i][0] += float(val)
                    countries[home_country][i][1] += 1

            for i, val in enumerate(away_features):
                if val != '':
                    countries[away_country][i][0] += float(val)
                    countries[away_country][i][1] += 1

        for country, value_list in countries.items():
            countries[country] = [round(val / max(1, count), 1) for [val, count] in value_list]

    return countries


def reconstruct(original_data: str, reconstructed_data: str) -> None:
    '''
    Reconstruct the missing data in original_data and write to new
    reconstructed_data csv.
    '''

    countries = get_averages(original_data)
    write_missing_data(original_data, reconstructed_data, countries)
    print([(c, avgs) for (c, avgs) in countries.items() if 0.0 in avgs])


if __name__ == '__main__':
    reconstruct('dataset/match_history.csv', 'reconstructed_match_history.csv')
