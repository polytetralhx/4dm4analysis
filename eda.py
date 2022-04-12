import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

four_dm_dataset = pd.read_csv('final_dataset.csv')

def get_tags(map_type, n):
    return [f'{map_type}{i+1}' for i in range(n)]

def get_map_labels(n_rc, n_hb, n_ln, n_sv):
    return get_tags('RC', n_rc) + get_tags('HB', n_hb) + get_tags('LN', n_ln) + get_tags('SV', n_sv) + get_tags('TB', 1)

map_label = {
    'Q': ['SV1', 'RC1', 'LN1', 'RC2', 'HB1'],
    'RO32': get_map_labels(5,2,3,2),
    'RO16': get_map_labels(5,2,3,2),
    'QF': get_map_labels(6,3,3,2),
    'SF': get_map_labels(6,3,3,2),
    'F': get_map_labels(7,3,4,2),
    'GF': get_map_labels(7,3,4,2)
}

# most data is skewed with the local maxima on the right
# so what the fuck should we do
# IDEA : REMOVE LEFT OUTLIER USING IQR THINGY idk
# NaN = ignore

def IQR(dataset: pd.Series):
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    return 1.5 * (Q3 - Q1)

def get_outlier(dataset: pd.Series):
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    iqr = IQR(dataset)

    return dataset[(dataset < (Q1 - iqr)) | (dataset > (Q3 + iqr))]

def logit_dataset(data: pd.DataFrame):
    d = data / 1000000
    return np.log(d / (1 - d))

def EDA(roundname):
    columns = list(filter(lambda x: x.split("_")[0] == roundname, four_dm_dataset.columns))
    round_column = logit_dataset(four_dm_dataset[columns])
    round_column = round_column.rename(columns={c: c.split("_")[-1] for c in columns})
    round_column.plot(kind='box', title=roundname + "_logit")

def plot_dist(data: pd.Series):
    data.plot(kind='hist')

if __name__ == "__main__":
    for round in map_label.keys():
        EDA(round)
        plt.savefig(round + ".png")

    plt.cla()
    plot_dist(data=logit_dataset(four_dm_dataset['QF_LN2']))
    plt.show() 

# what is next ? try to detect specialist according to round + pattern ? and use some of weighted stuff to get the probability of outlier