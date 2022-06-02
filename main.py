import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import f_regression
from dataset import Dataset
from models import LinearRegression, Ridge, Lasso, PolynomialRegression, PolyRidge, PolyLasso
from constants import *
from utils import generate_list_functions_from_generator_functions
from scipy.stats import shapiro

_4dm4 = Dataset('4dm4.db')

n_rounds = len(interested_rounds)

@generate_list_functions_from_generator_functions
def generate_models(polynomal_degrees: list, alpha_list: list):
    yield ('linearRegression', LinearRegression())
    for alpha in alpha_list:
        yield (f'ridge_alpha_{alpha}', Ridge(alpha))
        yield (f'lasso_alpha_{alpha}', Lasso(alpha))
    
    for degree in polynomal_degrees:
        yield (f'poly_{degree}', PolynomialRegression(degree))
        for alpha in alpha_list:
            yield (f'poly_{degree}_ridge_{alpha}', PolyRidge(degree, alpha))
            yield (f'poly_{degree}_lasso_{alpha}', PolyLasso(degree, alpha))

def plot_regression_model(model, label):
    a = np.arange(n_rounds).reshape((n_rounds, 1))
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
    print(category, "f-stat", float(f_stat), "p-value", float(p_value))

    degrees, alphas = [2,3], [0.5, 0.75, 1]

    models = generate_models(degrees, alphas)
    plt.figure()
    plt.title(category)
    plt.ylabel("ln(score/(1-score))")
    plt.xlabel("Round")
    plt.scatter(x, y)
    for model_label, model in models:
        model.fit(x, y)
        plot_regression_model(model, model_label)
        shapiro_stat, p_value_shapiro = shapiro(y - model.predict(x))
        print(model_label)
        print("Shapiro Wilk for Residuals")
        print("test stat", shapiro_stat, "p-value", p_value_shapiro)

    plt.legend()

regression_plot('RC')
plt.savefig('RC.png')
regression_plot('HB')
plt.savefig('HB.png')
regression_plot('LN')
plt.savefig('LN.png')
plt.show()
