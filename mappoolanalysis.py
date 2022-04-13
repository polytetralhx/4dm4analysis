import pandas as pd
import numpy as np
from scipy.stats import chi2, norm, t, f

t_cdf = t.cdf
t_invcdf = t.ppf

logit_dataset = pd.read_csv('4dm_logit.csv')


# some weird mappool analysis idk

# plan 
# Normal Distribution MODCHECK ?
# Mean Hypothesis testing with alpha = 0.05
# Variance test chi^2 something soemthing alpha = 0.05

def mean_hypothesis_test(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
    """
    Construct a hypothesis testing where null hypothesis is
    H0 : mu1 = mu2
    Ha : mu1 != mu2
    """
    std_ = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
    difference = mean1 - mean2
    df = ((std1 ** 2 / n1 + std2 ** 2 / n2) ** 2) / ((std1 ** 2 / n1) ** 2 / (n1 - 1) + (std2 ** 2 / n2) ** 2 / (n2 - 1))
    t_stat = difference / std_
    t_alpha = t_invcdf(alpha / 2, df)

    return t_cdf(-np.abs(t_stat), df), t_stat < t_alpha or t_stat > -t_alpha

def hypothesis_raw_dataset(data1: np.ndarray, data2: np.ndarray, alpha=0.05):
    data1 = data1[np.isnan(data1) == False]
    data2 = data2[np.isnan(data2) == False]
    mean1, mean2 = np.mean(data1), np.mean(data2)
    n1, n2 = len(data1), len(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    return mean_hypothesis_test(mean1, std1, n1, mean2, std2, n2)

def variance_hypothesis_test(s1, s2, n1, n2, alpha=0.05):
    """How to use scipy.stats.f lmao"""
    return 

def hypothesis_testing(round):
    all_maps_in_the_round = list(logit_dataset.columns)
    all_maps_in_the_round = list(filter(lambda x: x.split("_")[0] == round, all_maps_in_the_round))

    sus_pair = []
    for i, type1 in enumerate(all_maps_in_the_round):
        for j, type2 in enumerate(all_maps_in_the_round[i+1:]):
            data1 = logit_dataset[type1].values
            data2 = logit_dataset[type2].values
            if np.all(np.isnan(data1)) or np.all(np.isnan(data2)):
                continue
            p_value, reject_null = hypothesis_raw_dataset(data1, data2)
            print(type1, type2, p_value)
            if reject_null:
                sus_pair.append([type1, type2])

    return sus_pair


print(hypothesis_testing('SF'))