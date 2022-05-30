import pandas as pd
import numpy as np
from scipy.stats import t, f

t_cdf = t.cdf
t_invcdf = t.ppf
f_invcdf = f.ppf
f_cdf = f.cdf
f_inv_rt = f.isf


def mean_hypothesis_test(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
    """
    Construct a hypothesis testing where null hypothesis is
    H0 : mu1 = mu2
    Ha : mu1 != mu2
    """
    std_ = np.sqrt(std1**2 / n1 + std2**2 / n2)
    difference = mean1 - mean2
    df = ((std1**2 / n1 + std2**2 / n2) ** 2) / (
        (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1)
    )
    t_stat = difference / std_
    t_alpha = t_invcdf(alpha / 2, df)

    return t_cdf(-np.abs(t_stat), df), t_stat < t_alpha or t_stat > -t_alpha


def hypothesis_raw_dataset(data1: np.ndarray, data2: np.ndarray, alpha=0.05):
    data1 = data1[np.isnan(data1) == False]
    data2 = data2[np.isnan(data2) == False]
    mean1, mean2 = np.mean(data1), np.mean(data2)
    n1, n2 = len(data1), len(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    return mean_hypothesis_test(mean1, std1, n1, mean2, std2, n2, alpha)


def variance_hypothesis_test(s1, s2, n1, n2, alpha=0.05):
    """
    Construct a hypothesis testing where null hypothesis is
    H0 : sigma_1^2 = sigma_2^2
    Ha : sigma_1^2 != sigma_2^2
    """
    var_1 = s1**2
    var_2 = s2**2
    df1 = n1 - 1
    df2 = n2 - 1
    f_stat = var_1 / var_2
    f_alpha = f_inv_rt(alpha / 2, df1, df2)
    f_1_alpha = f_inv_rt(1 - alpha / 2, df1, df2)
    rj_null = not (f_1_alpha < f_stat and f_stat < f_alpha)
    return f_stat, rj_null


def variance_raw_dataset(data1: np.ndarray, data2: np.ndarray, alpha=0.05):
    data1 = data1[np.isnan(data1) == False]
    data2 = data2[np.isnan(data2) == False]
    n1, n2 = len(data1), len(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    return variance_hypothesis_test(std1, std2, n1, n2, alpha)
