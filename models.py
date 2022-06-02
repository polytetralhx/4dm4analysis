from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class PolynomialRegression():
    coef_ = None
    intercept_ = None
    def __init__(self, degree: int):
        self.model = Pipeline([('poly', PolynomialFeatures(degree, include_bias=False)), ('regression', LinearRegression())])
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.coef_ = self.model['regression'].coef_
        self.intercept_ = self.model['regression'].intercept_
    
    def predict(self, X):
        return self.model.predict(X)

class PolyRidge():
    coef_ = None
    intercept_ = None
    def __init__(self, degree: int, alpha: float):
        self.model = Pipeline([('poly', PolynomialFeatures(degree, include_bias=False)), ('regression', Ridge(alpha))])
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.coef_ = self.model['regression'].coef_
        self.intercept_ = self.model['regression'].intercept_
    
    def predict(self, X):
        return self.model.predict(X)
    
class PolyLasso():
    coef_ = None
    intercept_ = None
    def __init__(self, degree: int, alpha: float):
        self.model = Pipeline([('poly', PolynomialFeatures(degree, include_bias=False)), ('regression', Lasso(alpha))])

    def fit(self, X, y):
        self.model.fit(X, y)
        self.coef_ = self.model['regression'].coef_
        self.intercept_ = self.model['regression'].intercept_
    
    def predict(self, X):
        return self.model.predict(X)