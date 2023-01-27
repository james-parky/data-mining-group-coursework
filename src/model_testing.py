import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


def predict(training_data: str) -> None:
    dataset = pd.read_csv(training_data)
    dataset = dataset.replace(['Lose', 'Draw', 'Win'],[-1,0,1])
    dataset = dataset.select_dtypes([np.number])
    print(f'Original dataset columns: {list(dataset.columns)}')

    active_features = ['home_team_fifa_rank', 'away_team_fifa_rank',
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
    for column in dataset.columns:
        if column not in active_features:
            dataset = dataset.copy().drop([column], axis=1)
    print(f'Edited dataset columns: {list(dataset.columns)}')

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

    print(f'\033[96mLR Predictions: {np.mean(lr_predictions == y_test.values)*100} \033[0m')
    print(f'\033[96mSVM Predictions: {np.mean(svm_predictions == y_test.values)*100} \033[0m')
    print(f'\033[96mKNN Predictions: {np.mean(knn_predictions == y_test.values)*100} \033[0m')
    print(f'\033[96mMLP Predictions: {np.mean(mlp_predictions == y_test.values)*100} \033[0m')

    
    # Plotting confusion matrices
    lr_cm = confusion_matrix(y_test, lr_predictions, labels=lr_model.classes_)
    svm_cm = confusion_matrix(y_test, svm_predictions, labels=svm_model.classes_)
    knn_cm = confusion_matrix(y_test, knn_predictions, labels=knn_model.classes_)
    mlp_cm = confusion_matrix(y_test, mlp_predictions, labels=mlp_model.classes_)

    # Calculate f-score, precision & recall
    all_model_names = ["lr", "svm", "knn", "mlp"]
    all_predictions = [lr_predictions, svm_predictions, knn_predictions, mlp_predictions]
    for i in range(len(all_model_names)):
        print(all_model_names[i] + "F1 score: " + str(f1_score(y_test, all_predictions[i], average="macro")))
        print(all_model_names[i] + "Precision score: " + str(precision_score(y_test, all_predictions[i], average="macro")))
        print(all_model_names[i] + "Recall score: " + str(recall_score(y_test, all_predictions[i], average="macro")))


if __name__ == '__main__':
    print('\033[91mInterpolation no new features:\033[0m')
    predict('reconstructed_match_history_with_interpolation.csv')
    print('\033[91mLinear Regression no new features\033[0m')
    predict('reconstructed_with_lr_df.csv')
    print('\033[91mInterpoliation with new features:\033[0m')
    predict('dataset/interpolated_feature_match_history.csv')
    print('\033[91mLinear Regression with new features:\033[0m')
    predict('dataset/linear_feature_match_history.csv')
    print('\033[91mGaussian with new features:\033[0m')
    predict('dataset/gaussian_feature_match_history.csv')
    print('\033[91mInterpolation with new features and home_score included:\033[0m')
    predict('dataset/interpolated_feature_scores_match_history.csv')
    print('\033[91mLinear Regression with new features and home_score included:\033[0m')
    predict('dataset/linear_feature_scores_match_history.csv')
    print('\033[91mGaussian with new features and home_score included:\033[0m')
    predict('dataset/gaussian_feature_scores_match_history.csv')
