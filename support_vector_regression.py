'''
What is a Support Vector Machine?
    The objective of the support vector machine algorithm is to find a
    hyperplane in an N dimensional space that distinctly classifies the data
    points.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.pipeline import make_pipeline

def predict(training_data: str) -> None:
    dataset = pd.read_csv(training_data)
    dataset = dataset.replace(['Lose', 'Draw', 'Win'],[-1,0,1])
    # numerical_columns = [c for c in dataset.columns if dataset[c].dtype in [np.float64, np.int64]]
    dataset = dataset.select_dtypes([np.number])
    # test_set = test_set.select_dtypes([np.number])
    print(f'Original dataset columns: {list(dataset.columns)}')

    if training_data == 'dataset/match_history.csv':
        missing_rows = list(dataset.isnull().any(axis=1))
        for index, row in dataset.iterrows():
            if missing_rows[index] == True:
                dataset = dataset.drop(index)
    if training_data == 'reconstructed_match_history_with_interpolation.csv':
        dataset = dataset.copy().drop(['home_team_total_fifa_points', 'away_team_total_fifa_points', 'away_team_score'],axis=1)
    if training_data == 'reconstructed_with_lr_df.csv':
        dataset = dataset.copy().drop(['date'],axis=1)

    print(f'Edited dataset columns: {list(dataset.columns)}')
    #print(x_column_names)
    #x_data = dataset[['date,home_team,away_team,home_team_continent,away_team_continent,home_team_fifa_rank,away_team_fifa_rank,home_team_total_fifa_points,away_team_total_fifa_points,home_team_score,away_team_score,tournament,city,country,neutral_location,shoot_out,home_team_result,home_team_goalkeeper_score,away_team_goalkeeper_score,home_team_mean_defense_score,home_team_mean_offense_score,home_team_mean_midfield_score,away_team_mean_defense_score,away_team_mean_offense_score,away_team_mean_midfield_score']]
    
    from sklearn.linear_model import LogisticRegression
    #return np.mean(lr_predictions == y_test_lr.values)*100

    x_train, x_test, y_train, y_test = train_test_split(dataset.copy().drop(['home_team_result'],axis=1), dataset['home_team_result'], test_size=0.2)

    svm_model = make_pipeline(StandardScaler(), SVC(kernel='poly')).fit(x_train, y_train)
    svm_predictions = svm_model.predict(x_test)

    lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100000).fit(x_train, y_train)
    lr_predictions = lr_model.predict(x_test)

    from sklearn.neighbors import KNeighborsClassifier
    knn_model = KNeighborsClassifier(n_neighbors=40).fit(x_train, y_train)
    knn_predictions = knn_model.predict(x_test)

    from sklearn.neural_network import MLPClassifier
    mlp_model = MLPClassifier(random_state=1, max_iter=10000).fit(x_train, y_train)
    mlp_predictions = mlp_model.predict(x_test)

    print(f'LR Predictions: {np.mean(lr_predictions == y_test.values)*100}')
    print(f'SVM Predictions: {np.mean(svm_predictions == y_test.values)*100}')
    print(f'KNN Predictions: {np.mean(knn_predictions == y_test.values)*100}')
    print(f'MLP Predictions: {np.mean(mlp_predictions == y_test.values)*100}')

    
    '''    
    rows = []
    for index, row in dataset.iterrows():
        rows.append(row)
        for item in row:
            if item in [None, '']:
                print('missing data')
    '''
    #dataset['home_team_result'] = dataset['home_team_result'].replace(['Lose', 'Draw', 'Win'], [-1, 0, 1])

    #dataset.to_numpy()
    #print(dataset)


if __name__ == '__main__':
    print('Interpolation no new features:')
    predict('reconstructed_match_history_with_interpolation.csv')
    print('Linear Regression no new features')
    predict('reconstructed_with_lr_df.csv')
    print('Interpoliation with new features:')
    predict('dataset/interpolated_feature_match_history.csv')
    print('Linear Regression with new features:')
    predict('dataset/linear_feature_match_history.csv')
    print('Gaussian with new features:')
    predict('gaussian_feature_match_history.csv')
