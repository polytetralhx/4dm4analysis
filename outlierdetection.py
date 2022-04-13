import pandas as pd
import numpy as np
from eda import map_label

logit_dataset = pd.read_csv('4dm_logit.csv')

def get_outlier(round, map_type, iqrmultipiler=1.5):
    series = logit_dataset[round + "_" + map_type]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    left_ol = logit_dataset['player_name'][series < (q1 - iqrmultipiler * iqr)]
    right_ol = logit_dataset['player_name'][series > (q3 + iqrmultipiler * iqr)]

    return left_ol, right_ol

outlier_data = []
outlier_list = []
for round in map_label.keys():
    for map_type in map_label[round]:
        lbl = round + "_" + map_type
        left_ol, right_ol = get_outlier(round, map_type)
        for ol in left_ol:
            if ol in outlier_list:
                idx = outlier_list.index(ol)
                outlier_data[idx]['left_outlier'].append(lbl)
                continue
            outlier_data.append({
                'player_name': ol,
                'left_outlier': [lbl],
                'right_outlier': []
            })
            outlier_list.append(ol)
        
        for ol in right_ol:
            if ol in outlier_list:
                idx = outlier_list.index(ol)
                outlier_data[idx]['right_outlier'].append(lbl)
                continue
            outlier_data.append({
                'player_name': ol,
                'left_outlier': [],
                'right_outlier': [lbl]
            })
            outlier_list.append(ol)

outlier_dataframe = pd.DataFrame(outlier_data)
outlier_dataframe.to_csv('outliers.csv', index=False)