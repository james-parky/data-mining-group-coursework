'''
Code to reconstruct missing data in dataset/match_history.csv using average
score values for each country
'''
import csv
import matplotlib.pyplot as plt

HOME_FEATURE_INDEXES = [9, 17, 19, 20, 21]
AWAY_FEATURE_INDEXES = [10, 18, 22, 23, 24]
HOME_TEAM_FIFA_RANK = 5
AWAY_TEAM_FIFA_RANK = 6

def plot_fifa_score(original_data: str) -> None:
    '''
    Pull data from the given csv and plot 5 graphs, of score, offense score,
    defense score, midfield score and goalkeeper score against FIFA rank.
    '''
    with open(original_data) as csv_in_file:
        csv_reader = csv.reader(csv_in_file)
        next(csv_reader) # Skip the column headers

        ranks = []
        # features = [scores, gk_scores, def_scores, off_scores, mid_scores]
        features = [[], [], [], [], []]
        plot_colours = ['b','r','g','y','m']
        plot_titles = ['Score', 'Goalkeeper', 'Defense', 'Offense', 'Midfield']

        for row in csv_reader:
            # Extract the 5 types of score from the given csv for
            home_features = [row[i] for i in HOME_FEATURE_INDEXES]
            away_features = [row[i] for i in AWAY_FEATURE_INDEXES]

            # Only append the data to the lists is it is fully intact
            if '' not in home_features and '' not in away_features:
                ranks.append(int(row[HOME_TEAM_FIFA_RANK]))
                ranks.append(int(row[AWAY_TEAM_FIFA_RANK]))
                for i in range(5):
                    features[i].append(float(home_features[i]))
                    features[i].append(float(away_features[i]))

        # Create a figure with 5 subplots - one for each type of score
        fig, axs = plt.subplots(5,1, constrained_layout=True)
        fig.suptitle('A plot of different score metrics by FIFA ranking')

        # Zip the data, colour and title lists for the subplots
        plot_data = zip(plot_colours, plot_titles, features)
        for i, data in enumerate(plot_data):
            # For each score, plot it against the corresponding FIFA ranking
            axs[i].scatter(ranks, data[2], s=10, c=data[0], marker='.')
            axs[i].set_title(f'{data[1]} by FIFA rank')
            axs[i].set_xlabel('FIFA rank')
            axs[i].set_ylabel('Score')

        plt.show()



def write_missing_data(original_data: str, reconstructed_data: str,
                        countries: dict) -> None:
    '''
    For each row in the input csv, write to the output csv filling in missisng
    values, with that countries average for that feature.
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
    Find average values for each 5 features for each country.
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


if __name__ == '__main__':
    reconstruct('dataset/match_history.csv', 'reconstructed_match_history.csv')
    plot_fifa_score('dataset/match_history.csv')
