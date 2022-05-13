import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset import Dataset

_4DM_Dataset = Dataset('4dm4.db')
PLAYER_NAMES = _4DM_Dataset.select('player_data', ['player_name']).values.flatten()
ROUNDS = _4DM_Dataset.query('SELECT DISTINCT round FROM scores').values.flatten()
BEATMAP_TYPES = _4DM_Dataset.query('SELECT DISTINCT beatmap_type FROM scores').values.flatten()

interested_rounds = ['RO32', 'RO16', 'QF', 'SF', 'F', 'GF']
interested_beatmap_type = ['RC', 'HB', 'LN']

def format_sql_string(string_sql: str):
    return f"\"{string_sql}\""

def plot_all_data():
    for beatmap_type in interested_beatmap_type:
        plt.figure()
        plt.title(beatmap_type)
        for round in interested_rounds:
            tags = _4DM_Dataset.select(
                table='scores',
                columns=['beatmap_tag'],
                where={
                    'round': format_sql_string(round),
                    'beatmap_type': format_sql_string(beatmap_type),
                }
            ).values.flatten()
            
            tags = np.unique(tags)

            for tag in tags:
                scores = _4DM_Dataset.select(
                    table='scores', 
                    columns=['score_logit'], 
                    where={
                        'round': format_sql_string(round),
                        'beatmap_type': format_sql_string(beatmap_type),
                        'beatmap_tag': tag,
                    }
                ).values.flatten()
                
                plt.scatter([round] * len(scores), scores, color='b')

def plot_list_player_data(list_players: list):
    colors = ['b', 'r', 'g']
    fig, axs = plt.subplots(1, 3)
    for i, beatmap_type in enumerate(interested_beatmap_type):
        axs[i].set_title(beatmap_type)
        for round in interested_rounds:
            for j, player_name in enumerate(list_players):
                scores = _4DM_Dataset.select(
                    table='scores', 
                    columns=['score_logit'], 
                    where={
                        'round': format_sql_string(round),
                        'beatmap_type': format_sql_string(beatmap_type),
                        'player_name': format_sql_string(player_name),
                    }
                ).values.flatten()
                    
                # To be changed later idk
                axs[i].scatter([round] * len(scores), scores, label=player_name, color=colors[j])

plot_list_player_data(['shokoha', 'Micleak'])
plt.legend()
plt.show()