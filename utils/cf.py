import pandas as pd
import numpy as np


def vector_norm(a):
    return np.sqrt(np.sum(a**2))


def compare_vectors(a, b):
    return np.dot(a, b) / (vector_norm(a) * vector_norm(b))


def vector_intersect(a, b):
    not_nan_idx = np.where(np.logical_not(pd.isna(a) | pd.isna(b)))[0]
    return a[not_nan_idx], b[not_nan_idx]


def similarity(a: np.ndarray, b: np.ndarray):
    a_intersect, b_intersect = vector_intersect(a, b)
    if len(a_intersect) == 0:
        return 0
    return compare_vectors(a_intersect, b_intersect) * (len(a_intersect) / len(a))


class CollaborativeFiltering:
    def __init__(self, similarity_function=None):
        if similarity_function:
            self.similarity = similarity_function
        else:
            self.similarity = self.cosine_sim

    @staticmethod
    def cosine_sim(a, b):
        return similarity(a, b)

    def predict_(self, X: np.ndarray, idx: tuple):
        if not np.isnan(X[idx]):
            return X[idx]

        row, col = idx
        available_idx = np.where(np.logical_not(pd.isna(X[:, col])))
        available_data = X[available_idx]
        sum_abs_sim = 0
        sum_sim_data = 0
        for i in range(len(available_data)):
            sim = self.similarity(available_data[i], X[row])
            sum_abs_sim += np.abs(sim)
            sum_sim_data += sim * available_data[i][col]

        return sum_sim_data / sum_abs_sim

    def transform(self, X: np.ndarray):
        assert len(X.shape) == 2

        X_bar = np.nanmean(X, axis=1).reshape((X.shape[0], 1))
        X_minus_Xbar = X - X_bar
        row, col = X_minus_Xbar.shape
        X_predict = np.zeros((row, col))
        for i in range(row):
            for j in range(col):
                cf_predict = self.predict_(X_minus_Xbar, (i, j))
                X_predict[(i, j)] = cf_predict + X_bar[(i, 0)]

        return X_predict
