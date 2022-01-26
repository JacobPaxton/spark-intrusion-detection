import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import wrangle

def prep_model_MVP():
    '''
        Acquire the network intrusion dataset,
        Limit features to exploration-selected 'srv_count' and 'num_failed_logins',
        Fix column dtypes,
        Reduce DOS attack class rows by 95%,
        Group all attack classes into 'anomalous' class,
        Split data 50%-30%-20% for df, validate, and test splits,
        Isolate target column from each split,
        Scale features,
        Resample 'normal' and 'anomalous' class in train using SMOTE+Tomek,
        Return split / isolated data.
    '''
    # data ingest, prep, and split
    train, validate, test = wrangle.prep_model_MVP()

    # isolate target column in each split
    X_train, y_train = train.drop(columns='target'), train.target
    X_validate, y_validate = validate.drop(columns='target'), validate.target
    X_test, y_test = test.drop(columns='target'), test.target

    # scale each split
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # build resampler
    smtom = SMOTETomek(random_state=42)
    # resample the training split
    X_train_smtom, y_train_smtom = smtom.fit_resample(pd.DataFrame(X_train_scaled), y_train)

    # return splits
    return scaler, X_train_smtom, y_train_smtom, X_validate_scaled, y_validate, X_test_scaled, y_test


def bl_evaluation(y_train, y_validate):
    ''' Evaluate a baseline guess of 1 on train and validate, return reports '''
    # create, evaluate baseline for in-sample (train)
    bl_report_train = pd.DataFrame(
        classification_report(
            y_train,
            pd.Series([1 for _ in y_train]), 
            labels=[1, 0], 
            output_dict=True
        )
    ).T

    # create, evaluate baseline for out-of-sample (validate)
    bl_report_validate = pd.DataFrame(
        classification_report(
            y_validate,
            pd.Series([1 for _ in y_validate]), 
            labels=[1, 0], 
            output_dict=True
        )
    ).T

    return bl_report_train, bl_report_validate


def rf_evaluation(X_train, y_train, X_validate, y_validate):
    ''' Fit a random forest on train, evaluate on train and validate, return reports '''
    # build random forest classification model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # predict on train and validate
    rf_train_predictions = rf.predict(X_train)
    rf_validate_predictions = rf.predict(X_validate)

    # evaluate rf on train using classification report
    rf_report_train = pd.DataFrame(
        classification_report(
            y_train, 
            rf_train_predictions, 
            labels=[1, 0], 
            output_dict=True
        )
    ).T

    # evaluate rf on valudate using classification report
    rf_report_validate = pd.DataFrame(
        classification_report(
            y_validate, 
            rf_validate_predictions, 
            labels=[1, 0], 
            output_dict=True
        )
    ).T

    # return reports
    return rf, rf_report_train, rf_report_validate