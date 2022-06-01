import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset import Dataset
from utils import two_means_t_test, two_variances_f_test
from scipy.stats import t
from constants import interested_beatmap_type, interested_rounds

_4DM_Dataset = Dataset("4dm4.db")
imputed_4dm = Dataset("4dm4_impute.db")
PLAYER_NAMES = _4DM_Dataset.select("player_data", ["player_name"]).values.flatten()
ROUNDS = _4DM_Dataset.query("SELECT DISTINCT round FROM scores").values.flatten()
BEATMAP_TYPES = _4DM_Dataset.query(
    "SELECT DISTINCT beatmap_type FROM scores"
).values.flatten()


def format_sql_string(string_sql: str):
    return f'"{string_sql}"'


def plot_all_data():
    for beatmap_type in interested_beatmap_type:
        plt.figure()
        plt.title(beatmap_type)
        for round in interested_rounds:
            tags = _4DM_Dataset.select(
                table="scores",
                columns=["beatmap_tag"],
                where={
                    "round": format_sql_string(round),
                    "beatmap_type": format_sql_string(beatmap_type),
                },
            ).values.flatten()

            tags = np.unique(tags)

            for tag in tags:
                scores = _4DM_Dataset.select(
                    table="scores",
                    columns=["score_logit"],
                    where={
                        "round": format_sql_string(round),
                        "beatmap_type": format_sql_string(beatmap_type),
                        "beatmap_tag": tag,
                    },
                ).values.flatten()

                plt.scatter([round] * len(scores), scores, color="b")


def plot_list_player_data(list_players: list):
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9567bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    fig, axs = plt.subplots(1, 3)
    for i, beatmap_type in enumerate(interested_beatmap_type):
        axs[i].set_title(beatmap_type)
        for round in interested_rounds:
            for j, player_name in enumerate(list_players):
                scores = _4DM_Dataset.select(
                    table="scores",
                    columns=["score_logit"],
                    where={
                        "round": format_sql_string(round),
                        "beatmap_type": format_sql_string(beatmap_type),
                        "player_name": format_sql_string(player_name),
                    },
                ).values.flatten()
                # To be changed later idk
                axs[i].scatter(
                    [round] * len(scores), scores, label=player_name, color=colors[j]
                )


def get_CI(values, alpha):
    if len(values) < 2:
        return np.mean(values), np.mean(values)
    std = np.std(values, ddof=1)
    mean = np.mean(values)
    t_a = t.isf(alpha / 2, len(values) - 1)
    me = t_a * std / np.sqrt(len(values))
    return mean - me, mean + me


def plot_confidence_interval(
    dataset: Dataset, title: str, alpha: float, interested_players: list = []
):
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9567bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(title)
    for i, beatmap_type in enumerate(interested_beatmap_type):
        axs[i].set_title(beatmap_type)
        min_cis = []
        max_cis = []
        for round in interested_rounds:
            where = {
                "round": format_sql_string(round),
                "beatmap_type": format_sql_string(beatmap_type),
            }
            if len(interested_players) > 0:
                where["player_name"] = interested_players
            scores = dataset.select(
                table="scores", columns=["score * 0.000001 as score"], where=where
            ).values.flatten()

            min_ci, max_ci = get_CI(scores, alpha)
            min_cis.append(min_ci)
            max_cis.append(max_ci)
        axs[i].plot(
            interested_rounds,
            ([(a + b) / 2 for a, b in zip(min_cis, max_cis)]),
            color=colors[i],
        )
        axs[i].fill_between(
            interested_rounds, min_cis, max_cis, alpha=0.2, color=colors[i]
        )

def hypothesis_test_impute():
    for round in interested_rounds:
        for beatmap_type in interested_beatmap_type:
            base_where = {
                "round": format_sql_string(round),
                "beatmap_type": format_sql_string(beatmap_type),
            }
            original_data = imputed_4dm.select(
                "scores",
                ["score"],
                where={**base_where, "played": True},
            ).values.flatten()
            validated_data = imputed_4dm.select(
                "scores", ["score"], where=base_where
            ).values.flatten()
            p_value_t, reject_null_t = two_means_t_test(original_data, validated_data)
            p_value_f, reject_null_f = two_variances_f_test(
                original_data, validated_data
            )
            print(round, beatmap_type)
            print("mean ori / val :", np.mean(original_data), np.mean(validated_data))
            print(
                "var ori / val :",
                np.std(original_data, ddof=1),
                np.std(validated_data, ddof=1),
            )
            print("t", p_value_t, "f", p_value_f)


# imputed data vs original data
if __name__ == "__main__":
    plot_confidence_interval(_4DM_Dataset, "Original Data", 0.05)
    plot_confidence_interval(
        imputed_4dm, "Original Data + Missing Data Validation", 0.05
    )
    # hypothesis_test_impute()
    plt.show()
