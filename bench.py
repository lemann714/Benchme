#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
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
from argparse import ArgumentParser
from exception import ArgumentException
from settings import pdist
from pprint import pprint

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Random Forest",
    #"RBF SVM",
    #"Gaussian Process",
    #"Decision Tree",
    #"Neural Net",
    #"AdaBoost",
    "Naive Bayes",
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
    GaussianNB(),
    #QuadraticDiscriminantAnalysis(),
]
scoring = {
           'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'f1': make_scorer(f1_score),
           'roc_auc': make_scorer(roc_auc_score)
          }

def compare_classifiers(src: Path, target_index: int, sep: str) -> None:
    dataframe = pd.read_csv(src, sep=sep)#header=None, sep=sep)
    Y = dataframe.iloc[:, target_index].values
    X = dataframe.drop(dataframe.columns[target_index], axis=1).values
    #seed ensures that each classifier (clf) cross-validated in the same way
    cv_seed = 2022

    #url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    #col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    #dataframe = pd.read_csv(url, names=col_names)
    #array = dataframe.values
    #X = array[:,0:8]
    #Y = array[:,8]

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
    find_optimal_model(X, Y)

def find_optimal_model(data, target):
    metric = None
    avmetrics = list(scoring.keys())
    while metric not in avmetrics:
        try:
            metric = input(f"Choose metric from {avmetrics}. Press [Ctrl+c] to exit\n>>> ").strip()
        except KeyboardInterrupt:
            sys.exit(0)
    data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)
    best_score = 0.0
    best_params = None
    best_model = None
    index = None
    for i, clf in enumerate(classifiers):
        model_random_search = RandomizedSearchCV(clf,
                                                 param_distributions=pdist[names[i].strip().lower()],
                                                 n_iter=10,
                                                 cv=5,
                                                 scoring=metric,
                                                 verbose=1)
        model_random_search.fit(data_train, target_train)
        score = model_random_search.score(data_test, target_test)
        print(f"The test {metric} score of the best {names[i].upper()} model is {score:.2f}")
        if score > best_score:
            best_score = score
            best_model = names[i].lower()
            best_params = model_random_search.best_params_
            index = i
    print()
    print(f'Best model is {best_model.upper()}\n')
    # get the parameter names
    column_results = [f"param_{p}" for p in pdist[best_model].keys()]
    column_results += ["mean_test_score", "std_test_score", "rank_test_score"]
    cv_results = pd.DataFrame(model_random_search.cv_results_)
    cv_results = cv_results[column_results].sort_values("mean_test_score", ascending=False)
    cv_results = cv_results.rename(shorten_param, axis=1)
    cv_results = cv_results.set_index("rank_test_score")
    print(cv_results+'\n')
    print('Best parameters are:')
    pprint(best_params)

def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name

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

########################### argparser functionality part ###############################
cli = ArgumentParser(description='Performs benchmarking of binary classifiers on particular data')
cli.add_argument('source', type=str, help='CSV-data')
cli.add_argument('--target', '-t', type=int, default=-1, help="Index of target column")
cli.add_argument('--sep', '-s', type=str, default=',', help="Columns separator symbol")

if __name__ == '__main__':
    args = cli.parse_args()
    dags = vars(args)
    src = Path(dags['source'])
    tix = dags['target']
    sep = dags['sep']
    compare_classifiers(src, tix, sep)

