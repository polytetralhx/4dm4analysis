import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit

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
    
class LogisticCurveFitting():
    def __init__(self, p0):
        self.p0 = p0

    @staticmethod
    def logisticCurveFunction(x, x0, alpha, L, k):
        return (alpha + L / (1 - np.exp(-k * (x - x0))))
    
    def fit(self, x, y):
        self.p0, cov = curve_fit(self.logisticCurveFunction, x, y, self.p0)

        return cov
    
    def predict(self, x):
        return self.logisticCurveFunction(x, *self.p0)
    
    def min_max_possible_values(self):
        x0, alpha, L, k = self.p0
        
        return alpha, alpha + L
