import os
import matplotlib.pyplot as plt
from dataset import Dataset
from model import get_model
from constants import LOGIT_DATASET_DIR, FOURDM_DATASET_DIR, ROUNDS, INTERESTED_BEATMAP_TYPE

# player data + numeric data

LOGIT_DATASET = Dataset(LOGIT_DATASET_DIR).query(numeric=False)
FOURDM_DATASET = Dataset(FOURDM_DATASET_DIR).query(numeric=False)

# numeric data

LOGIT_NUMERIC = LOGIT_DATASET.query(numeric=True)
FOURDM_NUMERIC = FOURDM_DATASET.query(numeric=True)

# model for EDA / Skillbanning (KNN-Impute + PCA with dimensionality 2)

skillban_model = get_model(2)

if 'map_type' not in os.listdir():
    os.mkdir('map_type')

for map_type in INTERESTED_BEATMAP_TYPE:
    map_type_dataset = LOGIT_DATASET.query(beatmap_type=map_type)
    map_type_dataset.remove_unplayers()
    map_type_dataset = map_type_dataset.query(numeric=True)
    res = skillban_model.fit_transform(map_type_dataset.data.values)
    plt.figure()
    plt.title(map_type)
    plt.scatter(res[:, 0], res[:, 1])
    plt.savefig(f'map_type/{map_type}.png')

if 'rounds' not in os.listdir():
    os.mkdir('rounds')

for round in ROUNDS:
    map_type_dataset = LOGIT_DATASET.query(round=round)
    map_type_dataset.remove_unplayers()
    map_type_dataset = map_type_dataset.query(numeric=True)
    res = skillban_model.fit_transform(map_type_dataset.data.values)
    plt.figure()
    plt.title(round)
    plt.scatter(res[:, 0], res[:, 1])
    plt.savefig(f'rounds/{round}.png')


plt.show()
