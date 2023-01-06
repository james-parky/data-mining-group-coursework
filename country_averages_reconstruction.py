import csv


def reconstruct(original_data: str, reconstructed_data: str) -> None:
    countries = {}

    with open(original_data) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            home_country, away_country = row[1], row[2]
            #print(f'{home_country}, {away_country}')

            if home_country not in countries.keys():
                countries[home_country] = [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]] #[[float(row[9]), 1], [float(row[17]), 1] [float(row[19]), 1], [float(row[20]), 1], [float(row[21]), 1]]
            if away_country not in countries.keys():
                countries[away_country] = [[0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0], [0.0, 0]]
                # [(feature mean, times seen)]
            
            home_features = [row[9], row[17], row[19], row[20], row[21]]
            for i, f in enumerate(home_features):
                if f != '':
                    countries[home_country][i][0] += float(f)
                    countries[home_country][i][1] += 1

            away_features = [row[10], row[18], row[22], row[23], row[24]]
            for i, f in enumerate(away_features):
                if f != '':
                    countries[away_country][i][0] += float(f)
                    countries[away_country][i][1] += 1
    
        for country, value_list in countries.items():
            countries[country] = [round(val / max(1, count), 1) for [val,count] in value_list]
    
    with open(original_data) as csv_in_file:
        with open(reconstructed_data, 'w') as csv_out_file:
            csv_reader = csv.reader(csv_in_file)
            csv_writer = csv.writer(csv_out_file)
            csv_writer.writerow(next(csv_reader))

            for row in csv_reader:
                for i, val in enumerate(row):
                    if val == '':
                        if i in [9, 17, 19, 20, 21]:
                            replace_index = [9, 17, 19, 20, 21].index(i)
                            row[i] = countries[row[1]][replace_index]
                        elif i in [10, 18, 22, 23, 24]:
                            replace_index = [10, 18, 22, 23, 24].index(i)
                            row[i] = countries[row[2]][replace_index]
                csv_writer.writerow(row)


    print([(c, countries[c]) for c in countries.keys() if 0.0 in countries[c]])




if __name__ == '__main__':
    reconstruct('dataset/match_history.csv', 'reconstructed_match_history.csv')

# date,home_team,away_team,home_team_continent,away_team_continent,
# home_team_fifa_rank,away_team_fifa_rank,home_team_total_fifa_points,
# away_team_total_fifa_points,home_team_score,away_team_score,tournament,city,
# country,neutral_location,shoot_out,home_team_result,
# home_team_goalkeeper_score,away_team_goalkeeper_score,
# home_team_mean_defense_score,home_team_mean_offense_score,
# home_team_mean_midfield_score,away_team_mean_defense_score,
# away_team_mean_offense_score,away_team_mean_midfield_score

# NEEDED COLUMNS FOR HOME: 9, 17, 19, 20, 21
# NEEDED COLUMNS FOR AWAY: 10, 18, 22, 23, 24

# zambia missing goal keeper score and defense score for whole .csv