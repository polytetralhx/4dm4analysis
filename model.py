from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def get_model(pca_dim):
    return Pipeline([('imputer', KNNImputer()), ('pca', PCA(pca_dim))])


oneclassSVM = OneClassSVM()
kmeans = KMeans()
isolationforest = IsolationForest()
