from os import path
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib  # used for saving and loading the model
import json
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))
IGNORE_LOW_DATA = False  # ignore 102 instances with low score while training
BINARY_CLASSIFICATION = False  # do high vs low classification
CONVERT_SCORE_TO_INT = False  # must always be false as I am preconverting scores to int

feature_cols_to_use = [
    'Average_Speed',
    'Executor_Capacity',
    'Filled_Capacity',
    'Filled_Capacity_Percent',
    'Is_Sequence_Feasible',
    'Is_TSP_Feasible',
    'Is_Weekend',
    'Max_Segment_Dist',
    'Max_Segment_Time',
    'Num_Packages',
    'Num_Packages_Not_Delivered',
    'Num_Packages_TW',
    'Num_Stops',
    'Num_Stops_TW',
    'Num_Stops_TW_Violations',
    'Num_Zone_Switches',
    'Num_Zones',
    'Packages_Per_Stop',
    'Segment_Dist_Ratio',
    'Segment_Time_Ratio',
    'Slack_Ratio',
    'Switch_Stop_Ratio',
    'Switch_Zone_Ratio',
    'Total_Dist',
    'Total_End_Slack',
    'Total_Journey_Time',
    'Total_Max_Slack',
    'Total_Service_Time',
    'Total_Travel_Time',
    'Total_Wait_Time',
    'TSP_Optimality_Gap',
    'TSP_Route_Time',
    'Variance_Segment_Dist',
    'Variance_Segment_Time',
    'Vol_Packages_Not_Delivered',
    'Volume_Seconds_Traveled',
    'VST_Ratio',
    'Weighted_Pkg_End_Slack',
    'Weighted_Pkg_Max_Slack',
    'Weighted_Pkg_Slack_Ratio',
    'Weighted_Pkg_Wait_Time',
]


def get_all_training_data_med_high():
    """
    Gives only the rows with medium or high labels
    """
    route_summary_path = path.join(BASE_DIR, 'data/model_build_outputs/route_summary_full_jun18.csv')
    df = pd.read_csv(route_summary_path)
    df = df[~(df["Score"].str.contains("Low"))]
    # do not preconvert Score to categorical for neuralnetwork
    df["Score"] = df["Score"].astype("category")
    df.Score = pd.Categorical(df.Score, categories=["Medium", "High"], ordered=True)
    return df


def get_all_training_data():
    route_summary_path = path.join(BASE_DIR, 'data/model_build_outputs/route_summary_full_jun18.csv')
    df = pd.read_csv(route_summary_path)

    # order was always 0=high, 1=medium, 2=low; I wish to reverse it to be consistent
    df['intScore'] = 2
    df.loc[(df['Score'] == "Medium"), 'intScore'] = 1
    df.loc[(df['Score'] == "Low"), 'intScore'] = 0

    df['highOrNot'] = 0
    df.loc[~(df['Score'] == "High"), 'highOrNot'] = 1  # 0=High, 1=Not high

    # create dummy fields greater_than_low (gt_low), greater_than_med (gt_med) for ordinal classifiers
    df['gt_low'] = 0
    df.loc[~(df['Score'] == "Low"), 'gt_low'] = 1
    df['gt_med'] = 0
    df.loc[(df['Score'] == 'High'), 'gt_med'] = 1

    return df


def get_all_testing_data():
    test_summary_path = path.join(BASE_DIR, 'data/model_build_outputs/route_summary_full_jun18.csv')
    test_df = pd.read_csv(test_summary_path)

    # order was always 0=high, 1=medium, 2=low; I wish to reverse it to be consistent
    test_df['intScore'] = 2
    test_df.loc[(test_df['Score'] == "Medium"), 'intScore'] = 1
    test_df.loc[(test_df['Score'] == "Low"), 'intScore'] = 0

    test_df['highOrNot'] = 0
    test_df.loc[~(test_df['Score'] == "High"), 'highOrNot'] = 1  # 0=High, 1=Not high

    return test_df


def get_test_data_X_y():
    test_df = get_all_testing_data()
    # features sorted by name
    X = test_df[feature_cols_to_use]

    if not BINARY_CLASSIFICATION:
        y = test_df['intScore']
        if CONVERT_SCORE_TO_INT:
            y = pd.factorize(test_df['Score'])[0]  # convert y from Low, High, Med to integer
    else:
        y = test_df['highOrNot']
    return X, y


def get_training_data_X_y():
    if IGNORE_LOW_DATA:
        df = get_all_training_data_med_high()
    else:
        df = get_all_training_data()

    # features sorted by name
    X = df[feature_cols_to_use]

    if not BINARY_CLASSIFICATION:
        y = df['intScore']
        if CONVERT_SCORE_TO_INT:
            y = pd.factorize(df['Score'])[0]  # convert y from Low, High, Med to integer
    else:
        y = df['highOrNot']
    return X, y


def train_xgboost_classifier():
    dump_model = True

    from xgboost import XGBRegressor, XGBClassifier
    # from sklearn.model_selection import cross_val_score

    (X, y) = get_training_data_X_y()

    from sklearn.model_selection import train_test_split
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    eval_set = [(X, y), (X_test, y_test)]

    gbm = XGBClassifier(objective='multi:softmax',  
                        max_depth=6,
                        min_child_weight=1,
                        eta=0.2,
                        learning_rate=0.05,
                        n_estimators=200)
    gbm = gbm.fit(X, y, early_stopping_rounds=12, eval_metric="merror", eval_set=eval_set, verbose=True)

    # https://medium.com/swlh/xgboost-hyperparameters-optimization-with-scikit-learn-to-rank-top-20-44ea528efa58
    # parameters = {
    #     'max_depth': range(3, 8, 1),
    #     'min_child_weight': range(1, 5, 2),
    #     'n_estimators': range(100, 300, 50),
    #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
    #     'eta': [0.01, 0.05, 0.1]
    # }

    # gbm = GridSearchCV(estimator=XGBClassifier(
    #     use_label_encoder=False,
    #     gamma=0,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     objective='multi:softmax',
    #     nthread=4,
    #     scale_pos_weight=1,
    #     seed=18),
    #     param_grid=parameters,
    #     scoring='roc_auc',
    #     n_jobs=4,
    #     cv=5)
    # gbm.fit(X, y, early_stopping_rounds=12, eval_metric="merror", eval_set=eval_set, verbose=True)
    # print(gbm.best_estimator_)

    y_pred = gbm.predict(X_test)

    # should be in order
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    # print prob high
    df = get_all_testing_data()
    colNames = list(X_test.columns)

    high_samples = df.loc[df['Score'] == 'High', colNames]
    prob = gbm.predict_proba(high_samples)[:, 2]
    print("Average prob of high for high samples on entire dataset =", np.average(prob), "and other stats")
    print(stats.describe(prob))

    if dump_model:
        path_name = path.join(BASE_DIR, 'data/model_build_outputs/xgboost_compressed.joblib')
        joblib.dump(gbm, path_name, compress=3)
        # print(
        #     f"Compressed XGboost with size: {np.round(path.getsize('xgboost_compressed.joblib') / 1024 / 1024, 2) } MB")
