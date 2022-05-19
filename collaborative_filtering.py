import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from dataset import Dataset

interested_rounds = ['Q', 'RO32', 'RO16', 'QF', 'SF', 'F', 'GF']
interested_types = ['LN', 'HB', 'RC', 'TB']
_4dm = Dataset('4dm4.db')
old_ds = _4dm.get_old_dataset(interested_rounds, interested_types, True)
mean = _4dm.query(f"SELECT player_name, avg(score_logit) as average_score from scores where round in {tuple(interested_rounds)} and beatmap_type in {tuple(interested_types)} GROUP BY player_name")
old_ds = old_ds
for player_name in mean['player_name']:
    old_ds.loc[player_name] -= float(mean[mean['player_name'] == player_name]['average_score'])

knn = KNNImputer()
res = knn.fit_transform(old_ds)

new_ds = pd.DataFrame(res, index=old_ds.index, columns=old_ds.columns)
for player_name in mean['player_name']:
    new_ds.loc[player_name] += float(mean[mean['player_name'] == player_name]['average_score'])

new_ds = new_ds.apply(lambda x: 1000000 / (1 + np.exp(-x)))
new_ds.to_csv('impute2.csv')