from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def get_model(pca_dim):
    return Pipeline([('imputer', KNNImputer()), ('pca', PCA(pca_dim))])


oneclassSVM = OneClassSVM()
kmeans = KMeans(7)
isolationforest = IsolationForest()
lof = LocalOutlierFactor(n_neighbors=2)