import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

ACTIVE_FEATURES = ['home_team_fifa_rank', 'away_team_fifa_rank',
                       'home_team_result',
                       'home_team_goalkeeper_score',
                       'away_team_goalkeeper_score',
                       'home_team_mean_defense_score',
                       'home_team_mean_offense_score',
                       'home_team_mean_midfield_score',
                       'away_team_mean_defense_score',
                       'away_team_mean_offense_score',
                       'away_team_mean_midfield_score',
                       'home_team_total_average','away_team_total_average',
                       'goal_difference','home_team_rank_difference']

DATASETS = {
'INTERPOLATED_MATCH_HISTORY': '../dataset/reconstructed/interpolated_match_history.csv',
'LINEAR_REGRESSION_MATCH_HISTORY': '../dataset/reconstructed/linear_regression_match_history.csv',
'NEW_FEATURE_INTERPOLATED_MATCH_HISTORY': '../dataset/reconstructed/new_features_interpolated_match_history.csv',
'NEW_FEATURE_LINEAR_REGRESSION_MATCH_HISTORY': '../dataset/reconstructed/new_features_linear_regression_match_history.csv',
'NEW_FEATURE_GAUSSIAN_MATCH_HISTORY': '../dataset/reconstructed/new_features_gaussian_match_history.csv'}

def train_svm_model(x_train, y_train, x_test, y_test) -> float:
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='poly')).fit(x_train, y_train)
    svm_predictions = svm_model.predict(x_test)

    svm_cm = confusion_matrix(y_test, svm_predictions, labels=svm_model.classes_)
    return svm_predictions

def train_lr_model(x_train, y_train, x_test, y_test) -> float:
    lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100000).fit(x_train, y_train)
    lr_predictions = lr_model.predict(x_test)

    lr_cm = confusion_matrix(y_test, lr_predictions, labels=lr_model.classes_)
    return lr_predictions

def train_mlp_model(x_train, y_train, x_test, y_test) -> float:
    mlp_model = MLPClassifier(random_state=1, max_iter=10000).fit(x_train, y_train)
    mlp_predictions = mlp_model.predict(x_test)

    mlp_cm = confusion_matrix(y_test, mlp_predictions, labels=mlp_model.classes_)
    return mlp_predictions

def train_knn_model(x_train, y_train, x_test, y_test) -> float:
    knn_model = KNeighborsClassifier(n_neighbors=40).fit(x_train, y_train)
    knn_predictions = knn_model.predict(x_test)

    knn_cm = confusion_matrix(y_test, knn_predictions, labels=knn_model.classes_)
    return knn_predictions

def prep_data(file_name: str) -> tuple:
    dataset = pd.read_csv(file_name)
    dataset = dataset.replace(['Lose', 'Draw', 'Win'], [-1, 0, 1])
    dataset = dataset.select_dtypes([np.number])

    for column in dataset.columns:
        if column not in ACTIVE_FEATURES:
            dataset = dataset.copy().drop([column], axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(dataset.copy().drop(['home_team_result'],axis=1), dataset['home_team_result'], test_size=0.2)
    return (x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    for method, file_name in DATASETS.items():
        print(f'\033[91m{method}:\033[0m')
        x_train, x_test, y_train, y_test = prep_data(file_name)
        models = {'SVM': train_svm_model, 'LR': train_lr_model, 'KNN': train_knn_model, 'MLP': train_mlp_model}
        for model_type, func in models.items():
            predictions = func(x_train, y_train, x_test, y_test)
            print(f'\033[96m    {model_type} Predictions: {np.mean(predictions == y_test.values)*100} \033[0m')
            print(f'\033[93m        {model_type} F1 Score: {str(f1_score(y_test, predictions, average="macro", zero_division=0))}\033[0m')
            print(f'\033[93m        {model_type} Precision Score: {str(precision_score(y_test, predictions, average="macro", zero_division=0))}\033[0m')
            print(f'\033[93m        {model_type} Recall Score: {recall_score(y_test, predictions, average="macro", zero_division=0)}\033[0m')
        print('')
