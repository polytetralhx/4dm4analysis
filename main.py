import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_regression
from dataset import Dataset
from models import LinearRegression, Ridge, Lasso
from constants import *
from scipy.stats import shapiro

_4dm4 = Dataset('4dm4.db')

n_rounds = len(interested_rounds)

def plot_regression_model(model, label):
    a = [[0], [n_rounds]]
    plt.plot(a, model.predict(a), label=label)

def plot_residual(model, x, y_true):
    y_pred = model.predict(x)
    residual = y_true - y_pred
    plt.hist(residual)

def regression_plot(category: str):
    _data = _4dm4.get_label(interested_rounds, category, True)
    numeric_data = _data[['round_ord', 'avg_score_logit']]
    x, y = numeric_data['round_ord'], numeric_data['avg_score_logit']
    x = x.values.reshape(x.size, 1)

    f_stat, p_value = f_regression(x, y)
    print(category, "f-stat", f_stat, "p-value", p_value)

    # I am a "Statistician"
    linearRegression = LinearRegression()
    ridgeRegression_a_05 = Ridge(alpha=0.5)
    ridgeRegression_a_075 = Ridge(alpha=0.75)
    ridgeRegression_a_1 = Ridge(alpha=1)
    lassoRegression_a_05 = Lasso(alpha=0.5)
    lassoRegression_a_075 = Lasso(alpha=0.75)
    lassoRegression_a_1 = Lasso(alpha=1)

    # I "code" for research
    linearRegression.fit(x, y)
    ridgeRegression_a_05.fit(x, y)
    ridgeRegression_a_075.fit(x, y)
    ridgeRegression_a_1.fit(x, y)
    lassoRegression_a_05.fit(x, y)
    lassoRegression_a_075.fit(x, y)
    lassoRegression_a_1.fit(x, y)

    # Plotting Regression Models for comparison
    plt.figure()
    plt.title(category)
    plt.ylabel("ln(score/(1-score))")
    plt.xlabel("Round")
    plt.scatter(x, y)
    plot_regression_model(linearRegression, 'Linear Regression')
    plot_regression_model(ridgeRegression_a_05, 'Ridge alpha=0.5')
    plot_regression_model(ridgeRegression_a_075, 'Ridge alpha=0.75')
    plot_regression_model(ridgeRegression_a_1, 'Ridge alpha=1')
    plot_regression_model(lassoRegression_a_05, 'Lasso alpha=0.5')
    plot_regression_model(lassoRegression_a_075, 'Lasso alpha=0.75')
    plot_regression_model(lassoRegression_a_1, 'Lasso alpha=1')
    # shapiro test for committing Normal Distribution
    shapiro_stat, p_value_shapiro = shapiro(y - linearRegression.predict(x))
    print("Shapiro Wilk for Residuals")
    print("test stat", shapiro_stat, "p value", p_value_shapiro)
    plt.legend()

regression_plot('RC')
plt.savefig('RC.png')
regression_plot('HB')
plt.savefig('HB.png')
regression_plot('LN')
plt.savefig('LN.png')
plt.show()
