from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def get_model(pca_dim):
    return Pipeline([('imputer', KNNImputer()), ('pca', PCA(pca_dim))])