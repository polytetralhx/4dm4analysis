import os
import matplotlib.pyplot as plt
import numpy as np
import json
from dataset import Dataset
from model import get_model, isolationforest, kmeans, oneclassSVM, lof
from constants import LOGIT_DATASET_DIR, FOURDM_DATASET_DIR, ROUNDS, INTERESTED_BEATMAP_TYPE

# player data + numeric data

LOGIT_DATASET = Dataset(LOGIT_DATASET_DIR).query(numeric=False)
FOURDM_DATASET = Dataset(FOURDM_DATASET_DIR).query(numeric=False)

# numeric data

LOGIT_NUMERIC = LOGIT_DATASET.query(numeric=True)
FOURDM_NUMERIC = FOURDM_DATASET.query(numeric=True)

# model for EDA / Skillbanning (KNN-Impute + PCA with dimensionality 3)

skillban_model = get_model(3)

# SKILLBAN EDA

if 'map_type' not in os.listdir():
    os.mkdir('map_type')

pca_components = {}

for map_type in INTERESTED_BEATMAP_TYPE:
    map_type_dataset = LOGIT_DATASET.query(beatmap_type=map_type)
    map_type_dataset.remove_unplayers()
    map_type_dataset = map_type_dataset.query(numeric=True)
    res = skillban_model.fit_transform(map_type_dataset.data.values)
    plt.figure()
    plt.title(map_type)
    plt.scatter(res[:, 0], res[:, 1])
    plt.savefig(f'map_type/{map_type}.png')
    pca_components[map_type] = skillban_model['pca'].components_.copy().tolist()

if 'rounds' not in os.listdir():
    os.mkdir('rounds')

for round in ROUNDS:
    round_dataset = LOGIT_DATASET.query(round=round)
    round_dataset.remove_unplayers()
    round_dataset = round_dataset.query(numeric=True)
    res = skillban_model.fit_transform(round_dataset.data.values)
    plt.figure()
    plt.title(round)
    plt.scatter(res[:, 0], res[:, 1])
    plt.savefig(f'rounds/{round}.png')
    pca_components[round] = skillban_model['pca'].components_.copy().tolist()

with open('pca_components.json', 'w+') as f:
    json.dump(pca_components, f)
    f.close()

# Outlier Detection

def plot_outlier(players, pca_res, outlier_res, title, filename=None):
    plt.cla()
    plt.title(title)
    outlier_idx = np.where(outlier_res == -1)
    normal = np.where(outlier_res == 1)
    # plot normal data
    plt.scatter(pca_res[normal][:, 0], pca_res[normal][:, 1], color='b')
    # plot outlier with label
    plt.scatter(pca_res[outlier_idx][:, 0], pca_res[outlier_idx][:, 1], color='r')
    for outl in outlier_idx[0]:
        plt.annotate(players.values[outl], pca_res[outl, :2])
    if filename:
        plt.savefig(filename)
    
def plot_kmeans(players, pca_res, kmeans_res, title, filename=None, annotate=False):
    plt.cla()
    plt.title(title)

    for class_ in np.unique(kmeans_res):
        idx = np.where(kmeans_res == class_)
        plt.scatter(pca_res[idx][:, 0], pca_res[idx][:, 1])
        if annotate:
            for i in idx[0]:
                plt.annotate(players.values[i], pca_res[i, :2])
    
    if filename:
        plt.savefig(filename)

for map_type in INTERESTED_BEATMAP_TYPE:
    players, pca_res, outlier_res = LOGIT_DATASET.apply_outlier_model(None, map_type, skillban_model, oneclassSVM)
    plot_outlier(players, pca_res, outlier_res, f"oneclassSVM_{map_type}", f'oneclassSVM/{map_type}.png')
    players, pca_res, outlier_res = LOGIT_DATASET.apply_outlier_model(None, map_type, skillban_model, isolationforest)
    plot_outlier(players, pca_res, outlier_res, f"isolationForest_{map_type}", f'isolationForest/{map_type}.png')
    players, pca_res, outlier_res = LOGIT_DATASET.apply_outlier_model(None, map_type, skillban_model, lof)
    plot_outlier(players, pca_res, outlier_res, f"lof_{map_type}", f'lof/{map_type}.png')
    players, pca_res, outlier_res = LOGIT_DATASET.apply_outlier_model(None, map_type, skillban_model, kmeans)
    plot_kmeans(players, pca_res, outlier_res, f"kmeans_{map_type}", f'kmeans/{map_type}.png')

for round in ROUNDS:
    players, pca_res, outlier_res = LOGIT_DATASET.apply_outlier_model(round, None, skillban_model, oneclassSVM)
    plot_outlier(players, pca_res, outlier_res, f"oneclassSVM_{round}", f'oneclassSVM/{round}.png')
    players, pca_res, outlier_res = LOGIT_DATASET.apply_outlier_model(round, None, skillban_model, isolationforest)
    plot_outlier(players, pca_res, outlier_res, f"isolationForest_{round}", f'isolationForest/{round}.png')
    players, pca_res, outlier_res = LOGIT_DATASET.apply_outlier_model(round, None, skillban_model, lof)
    plot_outlier(players, pca_res, outlier_res, f"lof_{round}", f'lof/{round}.png')
    players, pca_res, outlier_res = LOGIT_DATASET.apply_outlier_model(round, None, skillban_model, kmeans)
    plot_kmeans(players, pca_res, outlier_res, f"kmeans_{round}", f'kmeans/{round}.png')

# with all outliers that has been collected
# how are we gonna select skillban players ?
# the answer might be handpicking but there should be some values for considerations
# isolationforest / kmeans looking good