from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class PolynomialRegression():
    def __init__(self, degree: int):
        self.model = Pipeline([('poly', PolynomialFeatures(degree)), ('regression', LinearRegression())])
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

class PolyRidge():
    def __init__(self, degree: int, alpha: float):
        self.model = Pipeline([('poly', PolynomialFeatures(degree)), ('regression', Ridge(alpha))])
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
class PolyLasso():
    def __init__(self, degree: int, alpha: float):
        self.model = Pipeline([('poly', PolynomialFeatures(degree)), ('regression', Lasso(alpha))])
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)