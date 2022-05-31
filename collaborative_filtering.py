import sqlite3
import numpy as np
import pandas as pd
from dataset import Dataset
from utils import CollaborativeFiltering
from utils import csv_to_sql

interested_rounds = ["Q", "RO32", "RO16", "QF", "SF", "F", "GF"]
interested_types = ["LN", "HB", "RC", "TB"]
_4dm_dataset = Dataset("4dm4.db")
csv_dataset = _4dm_dataset.get_old_dataset(interested_rounds, interested_types, True)
played = pd.notna(csv_dataset)
X = csv_dataset.values

cf_db = sqlite3.connect("4dm4_cf.db")
cf_cursor = cf_db.cursor()


# User based CF
cf = CollaborativeFiltering()

predict = cf.transform(X)
predict = 1000000 / (1 + np.exp(-predict))
predicted_value = pd.DataFrame(
    predict, index=csv_dataset.index, columns=csv_dataset.columns
)
predicted_value.to_csv("4dm4_user_cf_predict.csv")

# Item Based CF

predict_Item = cf.transform(X.T)
predict_Item = 1000000 / (1 + np.exp(-predict))
predict_Item = predict_Item.T
predict_Item_value = pd.DataFrame(
    predict, index=csv_dataset.index, columns=csv_dataset.columns
)
predict_Item_value.to_csv("4dm4_item_cf_predict.csv")
