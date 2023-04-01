"""
SIGNATE 【第32回_Beginner限定コンペ】診断データを使った糖尿病発症予測
https://signate.jp/competitions/748
このコードは上記のコンペで作成したコードです。
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import scipy.stats
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import catboost as cat
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import optuna
from tqdm import tqdm

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import torch
from itertools import combinations
import copy
BASE = os.getcwd() + os.sep
TARGET = "Outcome"

def main():
    global df_train
    df_train, df_test = read_file()
    # df_train = drop_outliers(df_train)
    # print(df_train)
    # study = optuna.create_study(direction='maximize')
    # study.optimize(xgb_objective, n_trials=500, show_progress_bar=True)
    # print(study.best_value)
    # print(study.best_params)
    """
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    all_model = get_models()
    val = 0
    record = []
    for i in range(1, len(all_model) + 1):
        for j in combinations(all_model, i):
            estimators = copy.deepcopy(j)
            clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter = 10 ** 5))
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            if val < score:
                val = score
                best_model = copy.deepcopy(estimators)
            model_name = []
            for k in j:
                model_name.append(k[0])
            print("score = ", score, "name = ", *model_name)
            record.append((score, *model_name))

    print(val, best_model)
    record.sort(reverse=True)
    tmp = pd.DataFrame(record)
    tmp.to_csv("result_2.csv", index=False)
    # X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET], random_state=42)
    """
    best_model = get_models()

    clf = StackingClassifier(estimators=best_model, final_estimator=LogisticRegression(max_iter=10 ** 6))
    # clf.fit(X_train, y_train)
    # print(clf.score(X_test, y_test))
    clf.fit(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    y_pred = clf.predict(df_test.drop(columns = "index"))
    #print(y_pred)
    # print(np.mean(cross_val_score(cat_model, X_train, y_train)))
    # model.fit(X_train, y_train)
    # model.fit(df_train.drop(columns = [TARGET, "index"]), df_train[TARGET])
    # y_pred = model.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    # cat_model 0.837245696400626
    # y_pred = model.predict(df_test.drop(columns = "index"))
    df_test["pred"] = y_pred
    df_test["pred"] = df_test["pred"].astype(int)
    print(df_test)
    sub = df_test[["index","pred"]]
    print(sub)
    sub.to_csv("submission.csv", index = False, header = False)
  

def read_file():
    train = pd.read_csv(BASE + "train.csv")
    test = pd.read_csv(BASE + "test.csv")
    return train, test

def drop_outliers(df_train):
    """
    外れ値対処
    """
    for i in df_train.columns:
        plt.plot(df_train[i])
        plt.show()
    np.random.seed(42)
    clf = IsolationForest()
    X = df_train.drop(columns=TARGET)
    clf.fit(X)
    predictions = clf.predict(X)
    df_train = df_train[predictions != -1]
    y = df_train[TARGET]
    X = df_train.drop(columns = TARGET)
    sm_enn = SMOTEENN(smote=SMOTE(k_neighbors=15), enn=EditedNearestNeighbours(n_neighbors=15))
    X_resampled, y_resampled = sm_enn.fit_resample(X, y)
    y_resampled = pd.DataFrame(y_resampled)
    # print(len(df_train[predictions != -1]), len(df_train))
    df_train = pd.concat([X_resampled, y_resampled], axis=1)
 
    return df_train

def lgb_objective(trial):
    """
    パラメータの最適化を行うための設定
    """
    params = {
        'n_iter': trial.suggest_int("n_iter", 50, 500),
        'verbosity': -1,
        'objective': trial.suggest_categorical("objective", ['binary', "cross_entropy"]),
        'extra_trees': True,
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.4, log=True),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-2, 10.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-2, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 8, 1024),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 250),
    }
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns= [TARGET, "index",]), df_train[TARGET])
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print(y_test, y_pred)
    return accuracy_score(y_test, y_pred)

def xgb_objective(trial):
    params = {
        'objective': 'binary:hinge',
        'tree_method': "hist",
        'n_estimators'  : trial.suggest_int('n_estimators', 100, 1000),                        
        'max_leaves' : trial.suggest_int('max_leaves', 2, 100),                                       
        'learning_rate' : trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample' : 0.50,
        'colsample_bytree': 0.50,
        'max_bin' : 4096,            
    }
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print(y_pred)
    return accuracy_score(y_test, y_pred)

def cat_objective(trial):
    params = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        'iterations' : trial.suggest_int('iterations', 50, 300),                         
        'depth' : trial.suggest_int('depth', 1, 15),                                       
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),               
        'random_strength' :trial.suggest_int('random_strength', 0, 100),                       
        'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), 
        'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        'od_wait' :trial.suggest_int('od_wait', 10, 50),
    }
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    model = cat.CatBoostClassifier(**params)
    model.fit(X_train, y_train, use_best_model=True)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
def logistic_objective(trial):
    params = {
        'tol' : trial.suggest_uniform('tol' , 1e-6 , 1e-3),
        'C' : trial.suggest_loguniform("C", 1e-2, 1),
       # 'fit_intercept' : trial.suggest_categorical('fit_intercept' , [True, False]),
       #  'random_state' : trial.suggest_categorical('random_state' , [0, 42, 2021, 555]),
       # 'solver' : trial.suggest_categorical('solver' , ['lbfgs','liblinear']),
        "n_jobs" : -1
    }
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def svc_objective(trial):
    param = {"C": trial.suggest_uniform("C", 0.1, 10),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid"]),
            "decision_function_shape": trial.suggest_categorical("decision_function_shape", ["ovo", "ovr"],)
    }

    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    model = SVC(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def dtc_objective(trial):
    param = {
        "criterion": 'gini',
        "max_depth": trial.suggest_int("max_depth", 1, 30),
        "max_features": trial.suggest_int("max_features", 1, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    model = DecisionTreeClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
def ada_objective(trial):
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1),
        "n_estimators": trial.suggest_int("n_estimators", 10, 300),
    }
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    model = AdaBoostClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def grad_objective(trial):
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1),
        "n_estimators": trial.suggest_int("n_estimators", 10, 300),
    }
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    model = GradientBoostingClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def random_forest_objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "criterion":  trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_depth": trial.suggest_int("max_depth", 1, 50),
        "min_samples_split" : trial.suggest_int("min_samples_split", 5, 50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
    }
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    model = RandomForestClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def extra_trees_objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "criterion":  trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 1, 50),
        "min_samples_split" : trial.suggest_int("min_samples_split", 5, 50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 2, 50),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
        "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0, 0.5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(columns=[TARGET, "index"]), df_train[TARGET])
    model = ExtraTreesClassifier(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def get_models():
    # 0.8528951486697965
    cat_params = {
     'objective': 'Logloss', 
     'iterations': 278, 
     'depth': 3, 
     'learning_rate': 0.17027783245133213, 
     'random_strength': 10, 
     'bagging_temperature': 1.9877222278568376, 
     'od_type': 'IncToDec', 
     'od_wait': 12,}
    cat_model = cat.CatBoostClassifier(**cat_params)
    # 0.8333333333333334
    lgb_params = {
        'n_iter': 313,
        'verbosity': -1,
        'objective': 'cross_entropy',
        'random_state': 42,
        'extra_trees': True,
         'colsample_bytree': 0.20655464850244626, 
         'colsample_bynode': 0.785611559656218, 
         'max_depth': 8, 
         'learning_rate': 0.16586739356801714, 
         'lambda_l1': 0.5346590839391885, 
         'lambda_l2': 8.785867530994723, 
         'num_leaves': 618, 
         'min_data_in_leaf': 7,
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    # 0.8426666666666667
    xgb_params = {
        'objective': 'binary:hinge',
        'tree_method': "hist",
        'n_estimators': 727, 
        'max_leaves': 61, 
        'learning_rate': 0.015181883588353672,               
    }
    xgb_model = xgb.XGBClassifier(**xgb_params)
    # 0.8093333333333333
    logistic_params = {'tol': 0.0003579020769427991, 'C': 0.2580833573708485}
    logistic_model = LogisticRegression(**logistic_params)
    # 0.7653333333333333
    knn_params = {'n_neighbors' : 6}
    knn_model = KNeighborsClassifier(**knn_params)
    # 0.8013333333333333
    svc_params = {'C': 4.568338387356583, 'kernel': 'rbf', 'decision_function_shape': 'ovr'}
    svc_model = SVC(**svc_params)
    # 0.808
    dtc_params = {'max_depth': 6, 'max_features': 10, 'min_samples_leaf': 8}
    dtc_model = DecisionTreeClassifier(**dtc_params)
    # 0.832
    ada_params = {'learning_rate': 0.28483757252281555, 'n_estimators': 273}
    ada_model = AdaBoostClassifier(**ada_params)
    # 0.8413333333333334
    grad_params = {'learning_rate': 0.09101263011302359, 'n_estimators': 109}
    grad_model = GradientBoostingClassifier(**grad_params)
    # 0.8346666666666667
    random_forest_params = {'n_estimators': 565, 'criterion': 'gini', 'max_depth': 16, 'min_samples_split': 49, 'max_leaf_nodes': 37, 'min_samples_leaf': 2}
    random_forest_model = RandomForestClassifier(**random_forest_params)
    #0.804
    extra_tree_params = {'n_estimators': 541, 'criterion': 'log_loss', 'max_depth': 7, 'min_samples_split': 10, 'max_leaf_nodes': 2, 'min_samples_leaf': 40, 'min_weight_fraction_leaf': 0.4290732153143506, 'max_features': 'sqrt'}
    extra_tree_model = ExtraTreesClassifier(**extra_tree_params)
    gaussNB_model = GaussianNB()

    estimators = [ ("ADa", ada_model), ("GB", grad_model), 
                  ("RF", random_forest_model),
                 ("CAT", cat_model)]
    all_models = [("cat", cat_model), ("lgb", lgb_model), ("xgb", xgb_model),
                  ("logi", logistic_model), ("knn", knn_model), ("svc", svc_model),
                  ("dtc", dtc_model), ("ada", ada_model), ("grad", grad_model), 
                  ("rf", random_forest_model), ("ext", extra_tree_model), ("NB", gaussNB_model),
                ]
    best_models = [("cat", cat_model), ("knn", knn_model), ("dtc", dtc_model),
                   ("ada", ada_model), ("rf", random_forest_model), ("NB", gaussNB_model),
                   ]
    model = VotingClassifier(estimators, n_jobs=-1)
    # return model
    score_best = [("ada", ada_model), ("grad", grad_model), ("rf", random_forest_model),
                  ("cat", cat_model),]
    return score_best
    # return best_models

if __name__ == "__main__":
    main()
