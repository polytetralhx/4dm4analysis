import pandas as pd

def format_values(dict_values):
    new_values = []
    for val in dict_values:
        if isinstance(val, str):
            new_values.append(f"\'{val}\'")
        else:
            new_values.append(str(val))
    return ", ".join(new_values)

def insert(cursor, table, data: dict):
    sql = f'INSERT INTO {table} ({", ".join(data.keys())}) values ({format_values(list(data.values()))})'
    cursor.execute(sql)

def csv_to_sql(cursor, full_dataset: pd.DataFrame, is_played: pd.DataFrame):
    index = full_dataset.index
    columns = full_dataset.columns

    for col in columns:
        round, map_type, map_tag = col.split("_")
        for idx in index:
            score = full_dataset.at[(idx, col)]
            played = is_played.at[(idx, col)]
            data = {
                'player_name': idx,
                'score': score,
                'played': int(played),
                'round': round,
                'beatmap_type': map_type,
                'beatmap_tag': map_tag
            }
            insert(cursor, 'scores', data)
