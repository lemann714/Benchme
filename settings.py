#!/usr/bin/env python

from scipy.stats import loguniform

class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)

pdist = dict()
pdist['nearest neighbors'] = {'n_neighbors': loguniform_int(1,10), 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': loguniform_int(1,10), 'p': [1,2]}
pdist['linear svm'] = {'C': loguniform(1,10), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': loguniform_int(1,10), 'gamma': ['scale', 'auto'], 'coef0': loguniform_int(1,10), 'tol': loguniform(1e-3, 1e-1), 'decision_function_shape': ['ovo', 'ovr']}

pdist['random forest'] = {'ccp_alpha': loguniform(1.0, 10.0), 'criterion': ['gini', 'entropy'], 'min_impurity_decrease': loguniform(1.0, 10.0), 'min_samples_leaf': loguniform_int(1, 10), 'min_samples_split': loguniform_int(2, 10), 'min_weight_fraction_leaf': loguniform(0.1, 0.49), 'n_estimators': loguniform_int(100,200)}
#pdist['random forest'] = {'n_estimators': loguniform_int(10,100), 'criterion': ['gini', 'entropy'], 'ccp_alpha': loguniform(1.0, 10.0)}

pdist['naive bayes'] = {'var_smoothing': loguniform(1e-9, 1e-5)}

pdist['decision tree'] = {'ccp_alpha': loguniform(1.0, 10.0), 'class_weight':[None], 'criterion': ['gini', 'entropy'], 'max_depth': [None], 'max_features': [None], 'max_leaf_nodes': [None], 'min_impurity_decrease': loguniform(1.0, 10.0), 'min_samples_leaf': loguniform_int(1, 10), 'min_samples_split': loguniform_int(2, 10), 'min_weight_fraction_leaf':loguniform(0.1, 0.49), 'random_state': [None], 'splitter': ['best']}
