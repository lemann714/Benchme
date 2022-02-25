#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from exception import ArgumentException
from pprint import pprint

class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)

pdist = dict()
pdist['nearest neighbors'] = {'n_neighbors': loguniform_int(1,100), 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': loguniform_int(1,100), 'p': [1,2]}
pdist['linear svm'] = {'C': loguniform(1,100), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': loguniform_int(1,10), 'gamma': ['scale', 'auto'], 'coef0': loguniform(1,100), 'shrinking': [True, False], 'tol': loguniform(1e-3, 1e-1), 'decision_function_shape': ['ovo', 'ovr']}
pdist['random forest'] = {'bootstrap': [True, False], 'ccp_alpha': loguniform(0.0, 10.0),'class_weight': ['balanced', 'balanced_subsample'], 'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2'], 'min_impurity_decrease': loguniform(0.0, 10.0), 'min_samples_leaf': loguniform_int(1, 10), 'min_samples_split': loguniform_int(1, 10), 'min_weight_fraction_leaf': loguniform(0.0, 10.0), 'n_estimators': loguniform_int(10,100), 'warm_start': [True, False]}

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Random Forest",
    #"RBF SVM",
    #"Gaussian Process",
    #"Decision Tree",
    #"Neural Net",
    #"AdaBoost",
    #"Naive Bayes",
    #"QDA"
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=5),
    #MLPClassifier(alpha=1, max_iter=1000),
    #AdaBoostClassifier(),
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis(),
]

def compare_classifiers():
    scoring = {
               'accuracy':make_scorer(accuracy_score),
               'precision':make_scorer(precision_score),
               'recall':make_scorer(recall_score),
               'f1_score':make_scorer(f1_score),
               'roc_auc': make_scorer(roc_auc_score)
              }

    # seed ensures that each classifier (clf) cross-validated in the same way
    cv_seed = 2022

    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pd.read_csv(url, names=col_names)
    array = dataframe.values
    X = array[:,0:8]
    Y = array[:,8]

    X = StandardScaler().fit_transform(X)

    dict_scores = defaultdict(list)
    for name, clf in zip(names, classifiers):
        print(f'Estimating {name} performance...')
        kfold = KFold(n_splits=7, shuffle=True, random_state=cv_seed)
        scores = cross_validate(clf, X, Y, cv=kfold, scoring=scoring)
        #clf.fit(x_train, y_train)
        #score = clf.score(x_test, y_test)
        dict_scores['Classifier'].append(name)
        for metric in scores.keys():
            dict_scores[metric].append(scores[metric].mean())
    table_scores = pd.DataFrame(dict_scores, columns=scores.keys())
    table_scores.insert(0, 'Classifiers', pd.Series(names, index=table_scores.index))
    print(table_scores)
    sort_df(table_scores)
    #try:
    #    clf = input("Parameters tuning. Enter classifier:\n>>> ")
    #except KeyboardInterrupt:
    #    sys.exit(0)
    find_optimal_model(data)

def find_optimal_model(data):
    metric = None
    avmetrics = list(scoring.keys())
    while metric not in avmetrics:
        try:
            metric = input(f"Choose metric from {avmetrics}. Press [Ctrl+c] to exit\n>>> ").strip()
        except KeyboardInterrupt:
            sys.exit(0)
    best_score = 0.0
    best_params = None
    best_model = None
    index = None
    for i, clf in enumerate(classifiers):
        clf = classifiers[names.index(clf_name)]
        model_random_search = RandomizedSearchCV(clf,
                                                 param_distributions=pdist[clf_name.strip().lower()],
                                                 n_iter=10,
                                                 cv=5,
                                                 scoring=scoring,
                                                 verbose=1)
        data_train, data_test, target_train, target_test = train_test_split(
                                                 data, target, random_state=42)
        model_random_search.fit(data_train, target_train)
        score = model_random_search.score(data_test, target_test)
        print(f"The test {metric} score of the best {names[i].upper()} model is {score:.2f}")
        if score > best_score:
            best_score = score
            best_model = names[i]
            best_params = model_random_search.best_params_
            index = i
    print(f'Best model: {best_model}')
    print('Best parameters:')
    pprint(best_params)

def sort_df(df):
    def read():
        try:
            sort_column = input("Enter column name to order classifiers. Press [p] to optimal model search, [Ctrl+c] to exit:\n>>> ")
        except KeyboardInterrupt:
            sys.exit(0)
        return sort_column

    scol = read()
    while scol.strip() != 'p':
        try:
            df.sort_values(by=scol, inplace=True, ascending=True)
            print(df)
            print()
        except KeyError:
            print(f'Wrong column name. Choose from {list(df.columns)}')
        scol = read()
'''
def fine_tune(data, target, clf_name, scoring):
    clf = classifiers[names.index(clf_name)]
    model_random_search = RandomizedSearchCV(clf,
                                             param_distributions=pdist[clf_name.strip().lower()],
                                             n_iter=10,
                                             cv=5,
                                             scoring=scoring,
                                             verbose=1)
    data_train, data_test, target_train, target_test = train_test_split(
                                             data, target, random_state=42)
    model_random_search.fit(data_train, target_train)
    accuracy = model_random_search.score(data_test, target_test)
    print(f"The test accuracy score of the best model is {accuracy:.2f}")
'''
if __name__ == '__main__':
    #if sys.argv != 2:
    #    raise ArgumentException("Source file must be given as an argument")
    #source = sys.argv[1]
    compare_classifiers()

