import pandas as pd

def get_table(dataframe: pd.DataFrame):
    players = dataframe['country_name'].unique()
    rounds = dataframe['round'].unique()
    beatmap_type = dataframe['beatmap_type'].unique()
    table_dict = {}
    for player in players:
        player_dict = {}
        for round in rounds:
            for cat in beatmap_type:
                score_player_round = dataframe[(dataframe['country_name'] == player) & (dataframe['round'] == round) & (dataframe['beatmap_type'] == cat)][['beatmap_tag', 'score_logit']]
                for beatmap_tag, score_logit in score_player_round.values:
                    player_dict["_".join([round, cat, str(int(beatmap_tag))])] = score_logit

        table_dict[player] = player_dict
    
    return pd.DataFrame(table_dict).transpose()