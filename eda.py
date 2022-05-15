import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset import Dataset
from scipy.stats import t

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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9567bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
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

def get_CI(values, alpha):
    if len(values) < 2:
        return np.mean(values), np.mean(values)
    std = np.std(values, ddof=1)
    mean = np.mean(values)
    t_a = t.isf(alpha / 2, len(values) - 1)
    me = t_a * std / np.sqrt(len(values))
    # WYSI
    return mean - me, mean + me

def plot_confidence_interval(alpha, interested_players: list):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9567bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig, axs = plt.subplots(1, 3)
    fig.suptitle("95% Confidence Interval of Average Scores of 4dm4 Grand Finalists")
    for i, beatmap_type in enumerate(interested_beatmap_type):
        axs[i].set_title(beatmap_type)
        min_cis = []
        max_cis = []
        for round in interested_rounds:
            scores = _4DM_Dataset.select(
                table='scores',
                columns=['score'],
                where={
                    'round': format_sql_string(round),
                    'beatmap_type': format_sql_string(beatmap_type),
                    'player_name': interested_players
                }
            ).values.flatten()

            min_ci, max_ci = get_CI(scores, alpha)
            min_cis.append(min_ci)
            max_cis.append(max_ci)
        axs[i].plot(interested_rounds, ([(a + b) / 2 for a, b in zip(min_cis, max_cis)]), color=colors[i])
        axs[i].fill_between(interested_rounds, min_cis, max_cis, alpha=0.2, color=colors[i])

grand_finalists = np.unique(_4DM_Dataset.select('scores', ['player_name'], {
    'round': format_sql_string('GF')
}).values.flatten()).tolist()

plot_confidence_interval(0.05, grand_finalists)
plt.legend()
plt.show()