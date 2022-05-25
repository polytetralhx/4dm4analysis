import sqlite3
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from dataset import Dataset

interested_rounds = ["Q", "RO32", "RO16", "QF", "SF", "F", "GF"]
interested_types = ["LN", "HB", "RC", "TB"]
_4dm = Dataset("4dm4.db")
old_ds = _4dm.get_old_dataset(interested_rounds, interested_types, True)
played = pd.notna(old_ds)
mean = _4dm.query(
    f"SELECT player_name, avg(score_logit) as average_score from scores where round in {tuple(interested_rounds)} and beatmap_type in {tuple(interested_types)} GROUP BY player_name"
)
old_ds = old_ds
for player_name in mean["player_name"]:
    old_ds.loc[player_name] -= float(
        mean[mean["player_name"] == player_name]["average_score"]
    )

knn = KNNImputer(n_neighbors=2)
res = knn.fit_transform(old_ds)

new_ds = pd.DataFrame(res, index=old_ds.index, columns=old_ds.columns)
for player_name in mean["player_name"]:
    new_ds.loc[player_name] += float(
        mean[mean["player_name"] == player_name]["average_score"]
    )

new_ds = new_ds.apply(lambda x: 1000000 / (1 + np.exp(-x)))
new_sql = sqlite3.connect("4dm4_impute.db")
cursor = new_sql.cursor()


def format_values(dict_values):
    new_values = []
    for val in dict_values:
        if isinstance(val, str):
            new_values.append(f"'{val}'")
        else:
            new_values.append(str(val))
    return ", ".join(new_values)


def insert(table, data: dict):
    sql = f'INSERT INTO {table} ({", ".join(data.keys())}) values ({format_values(list(data.values()))})'
    cursor.execute(sql)


def impute_data_to_sql(full_dataset: pd.DataFrame, is_played: pd.DataFrame):
    index = full_dataset.index
    columns = full_dataset.columns

    for col in columns:
        round, map_type, map_tag = col.split("_")
        for idx in index:
            score = full_dataset.at[(idx, col)]
            played = is_played.at[(idx, col)]
            data = {
                "player_name": idx,
                "score": score,
                "played": int(played),
                "round": round,
                "beatmap_type": map_type,
                "beatmap_tag": map_tag,
            }
            insert("scores", data)


impute_data_to_sql(new_ds, played)
new_sql.commit()
