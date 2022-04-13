import pandas as pd
import numpy as np
from scipy.stats import chi2, norm, t, f

chi2_invcdf = chi2.isf
norm_invcdf = norm.ppf
norm_cdf = norm.cdf
f_invcdf = f.ppf
f_cdf = f.cdf

logit_dataset = pd.read_csv('4dm_logit.csv')

"""Ignore the code here it's just my failed attempt on Homogenity test"""

# def distribute(seq: np.ndarray, ranges: np.ndarray):
#     distributed_data = []
#     for i, j in zip(ranges[:-1], ranges[1:]):
#         distributed_data.append(len(seq[(i <= seq) & (seq < j)]))
#     return np.array(distributed_data) / len(seq)

# def test_same_distribution(round1: str, type1: str, round2: str, type2: str, alpha: float, n_freq: int = 20):
#     col1 = round1 + "_" + type1
#     col2 = round2 + "_" + type2
#     seq1 = logit_dataset[col1]
#     seq1 = seq1[pd.isna(seq1) == False].values
#     seq2 = logit_dataset[col2]
#     seq2 = seq2[pd.isna(seq2) == False].values
#     # standardize
#     std_seq1 = (seq1 - np.mean(seq1)) / np.std(seq1)
#     std_seq2 = (seq2 - np.mean(seq2)) / np.std(seq2)
    
#     all_data = np.concatenate((std_seq1, std_seq2))
#     ranges = np.linspace(np.min(all_data), np.max(all_data), n_freq)
#     df = n_freq - 1

#     distributed_seq1 = distribute(std_seq1, ranges)
#     distributed_seq2 = distribute(std_seq2, ranges)
#     distributed_allseq = distribute(all_data, ranges)

#     x2_stat = np.sum(((distributed_seq1 - distributed_allseq) ** 2) / distributed_allseq) + np.sum(((distributed_seq2 - distributed_allseq) ** 2) / distributed_allseq)
#     x2_alpha = chi2_invcdf(alpha, df)

#     return x2_stat, x2_stat > x2_alpha

# print(test_same_distribution('Q', 'RC1', 'F', 'HB3', 0.05, 5))

