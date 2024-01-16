import numpy as np

import pandas as pd

from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest, SequentialFeatureSelector
from sklearn.metrics import ConfusionMatrixDisplay

from feature_extractor import feature_extractor_folder
from data import FeatureTypes, Frame, MaskValues

import matplotlib.pyplot as plt


def get_data():
    X_df = pd.read_csv("./data/features.csv", header=0, index_col=0, sep=";")
    Y = pd.read_csv("./data/groups.csv", header=None, index_col=0, sep=";").values

    X_train = X_df.values.T[:100, :]
    X_test = X_df.values.T[100:, :]

    Y_train = Y[:100].ravel()
    Y_test = Y[100:].ravel()

    SS = StandardScaler()
    SS.fit(X_train)
    X_train = SS.transform(X_train)
    X_test = SS.transform(X_test)

    LE = LabelEncoder()
    LE.fit(Y_train)
    Y_train = LE.transform(Y_train)
    Y_test = LE.transform(Y_test)

    return X_train, Y_train, X_test, Y_test


def get_svm():
    SVM = LinearSVC(C=10, penalty='l2', loss='hinge', dual=True, max_iter=1000000)
    return SVM

def pca(n_features):
    X_train, Y_train, X_test, Y_test = get_data()
    SVM = get_svm()

    pca = PCA(n_components=n_features)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    SVM.fit(X_train, Y_train)
    scores = SVM.score(X_test, Y_test)
    return scores


def mutual_info(n_features):
    X_train, Y_train, X_test, Y_test = get_data()
    SVM = get_svm()

    importances = mutual_info_classif(X_train, Y_train)
    kept_indices = [i for _,i in sorted(zip(importances, list(range(len(importances)))))[::-1][:n_features]]
    X_train = X_train[:, kept_indices]
    X_test = X_test[:, kept_indices]

    SVM.fit(X_train, Y_train)
    scores = SVM.score(X_test, Y_test)
    return scores


def sfs(n_features):
    X_train, Y_train, X_test, Y_test = get_data()
    SVM = get_svm()

    loo = LeaveOneOut()
    sfs = SequentialFeatureSelector(SVM, n_features_to_select=n_features, direction="forward", n_jobs=4, cv=5)
    sfs.fit(X_train, Y_train)
    X_train = sfs.transform(X_train)
    X_test = sfs.transform(X_test)

    SVM.fit(X_train, Y_train)
    scores = SVM.score(X_test, Y_test)
    return scores

if __name__ == "__main__":
    nb_features = 20
    print(sfs(nb_features))
