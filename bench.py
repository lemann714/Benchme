#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
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
#pd.set_option('display.max_columns', None)

def compare_classifiers():
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        #"RBF SVM",
        #"Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA"
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        #SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

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

def sort_df(df):
    def read():
        try:
            sort_metric = input("Enter metric to order classifiers [p to proceed, Ctrl+c to exit]: ")
        except KeyboardInterrupt:
            sys.exit(0)
        return sort_metric

    while read() != 'p':
        try:
            df.sort_values(by=sort_metric, ascending=False)
            print(df)
        except KeyError:
            print('Unknown metric')

if __name__ == '__main__':
    #if sys.argv != 2:
    #    raise ArgumentException("Source file must be given as an argument")
    #source = sys.argv[1]
    compare_classifiers()

