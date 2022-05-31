# Outlier Detection for Rank Restricted Rhythm Game Tournaments

## Project Description

This is a project for analysis of [4 Digit MWC 4](https://osu.ppy.sh/wiki/en/Tournaments/4DM/4) Player Scores from Qualifiers to Grand Finals. This will include the Exploratory Data Analysis, Predicting the missing scores and Outlier Detection (or in other words, `Skillbanning`).

---

## Important Definitions

`4 digits` - An osu!mania player who has the 4k rank of 1000 to 9999

`Derankers` - A 4 digits whose skill exceeds the 4 digits skill level

`Skillbanning` - A method to select or classify the Derankers

`Beatmap Type` - A type of a beatmap, in the dataset it's labelled as `RC`, `HB`, `LN`, `SV` and `TB`. In this study we will omit the `SV` since it is not a physical skill

---

## Limitations

This project will limit to only 4dm4 Tournament Scores from the Official Multiplayer Links. The data was collected in the statistics sheet and it can be found [here](https://docs.google.com/spreadsheets/d/1ahkEH9dOcpeAWHYfpUOKo_0AjD_aIOYzmsa8fKfGBAs/edit?rm=minimal#gid=254546040). However the dataset will be uploaded on Kaggle soon if HowToPlayLN is not lazy.

This project will soon expand to ranked and loved maps but after we finish the analysis on the tournament scores.

---

## Methodology

### Exploratory Data Analysis

TBA

### Outlier Detection Model

TBA

### Model Evaluation

TBA

#### Hand-Picking Outliers

TBA

#### Evaluation Metric

<div align="center"><em>"Even if statistical skillban is possible, there will still be the outliers"</em></div>

After completing the process of Hand-Picking, we use the method of Precision and Recall ([Bohutska, 2021](https://towardsdatascience.com/anomaly-detection-how-to-tell-good-performance-from-bad-b57116d71a10)). We compare the model prediction with the hand-picked outliers using the formula for

- Recall : The true positive rate (the result agrees each other) compared with the number of hand-picked outliers

![](https://miro.medium.com/max/604/1*JFFXyQmWC7b-HBR8Xp-o-Q.png)

- Precision : The true positive rate (the result agree each other) compared with the number of model prediction

![](https://miro.medium.com/max/588/1*4I2sVOt7CBoR2bOAugBnYw.png)

We can combine them together using `F1 Score` formula

![](https://miro.medium.com/max/428/1*ywQ6BojJMPImBLkLv_jv2Q.png)

However, we need to consider the `False Positive Rate` and `False Negative Rate` (Pardon my english a bit lol)

- False Positive Rate : The possibility of model prediction is an outlier where the hand-picked does not agree
- False Negative Rate : The possibility of model prediction is not an outlier where the hand-picked does not agree

The model with High `F1 Score` and satisfying `False Positive Rate` and `False Negative Rate` will be considered as a suitable model to detect the outliers.

#### Testing

TBA

---

## Code Documentation

### Dataset Class

There is a `Dataset` class inside the `dataset.py` file. Which is made to deal with `pandas.DataFrame` and `SQLite3 Database` (specially for 4dm4 players / scores data) there are 3 important methods in this class (and potentially more methods)

#### Initialization

The `Dataset` class can be initialized using the sqlite `.db` file

```python {all|2|1-6|9|all}
from dataset import Dataset

_4dm4 = Dataset('4dm4.db')
```

#### query

The `Dataset().query` method is implemented for SQLite Query in the `.db` file and output type `pandas.DataFrame` of the query result.

```python {all|2|1-6|9|all}
query_result = _4dm4.query("SELECT * FROM scores")
query_result # pandas.DataFrame
```

#### select

The `Dataset().select` method is the extension of `Dataset().query` method.
It is a method to select the data from table: **table**, returns all data from table if **columns** and **where** is not provided

**Example 1** : Select with conditional filtering

```python {all|2|1-6|9|all}
# Select player_name, beatmap_type, beatmap_tag, score, score_logit from scores
# where score > 990k in Qualifiers Round
ds = dataset.select(
    table='scores',
    columns=['player_name', 'beatmap_type', 'beatmap_tag', 'score', 'score_logit'],
    where={
        'score': ">990000",
        'round': "\"Q\""
    }
)
```

**Example 2** : Select without column provided

```python {all|2|1-6|9|all}
# Select all columns from scores
# where score > 990k
ds = dataset.select(
    table='scores',
    columns=['player_name', 'beatmap_type', 'beatmap_tag', 'score', 'score_logit'],
    where={
        'score': ">990000"
    }
)
```

#### get_old_dataset

The `Dataset().get_old_dataset` is for returning the `pandas.DataFrame` of `scores` table with columns being the beatmaps and indecies being the `players`. This will contain the null data if it is used with `4dm4.db`. This method is for validating data using Collaborative Filtering or KNN Imputation (as they are easier to manage).

### Utility Functions

Utility Functions are in the module named `utils`. It is implemented especially for this study.

#### Collaborative Filtering

The `Adjusted Collaborative Filtering` is used for missing score validation and it is in the class called `CollaborativeFiltering`. The example of how to use can be found here.

```python {all|2|1-6|9|all}
import numpy as np
from utils import CollaborativeFiltering

a = np.array(
    [
        [1,np.nan,2],
        [2,3,np.nan],
        [np.nan,4,4]
    ]
)
]

# Implement Adjusted Collaborative Filtering Model
cf = CollaborativeFiltering()
validated_data = cf.transform(a)
validated_data # expect to return np.array without null data
```

#### csv to sqlite database

In order to transform from validated 2d pandas array from `Collaborative Filtering` or `sklearn.KNNImputer` to sqlite database, we use `utils.csv_to_sql` function to do that.

In order to use this function, we need `sqlite3.Connection` data type to do that (from `sqlite3.connect` function) and `played` dataset.

The example can be seen in `knnimpute.py`

#### Hypothesis Testing (t-test and f-test)

In order to compare the validated data with original data, we use `t-test` (will consider changing to `nonparametric test` later after index learns stuff) for comparing means (averages) and `f-test` for comparing variances. These are in `utils` module and their names are `two_means_t_test` and `two_variances_f_test`. The inputs are two `np.ndarrays` (of two datasets) and `alpha` (default `0.05`) and the outputs are `p-value` and `reject_null`.

```python
import numpy as np
from utils import two_means_t_test, two_variances_f_test

dataset1 = np.random.normal(0, 1, 100)
dataset2 = np.random.normal(1, 0.75, 120)

# perform two tails t-test for means with alpha=0.05
p_value_mean, reject_null_mean = two_means_t_test(dataset1, dataset2, alpha=0.05)
# perform two tails f-test for vairances with alpha=0.05
p_value_variance, reject_null_variance = two_variances_f_test(
    dataset1, dataset2, alpha=0.05
)

```

---

## Current Problems / Challenges

TBA

---

## Editor's Notes

### Dataset, Visualizations and Code Archives (Failed Attempts that are not included in documentation)

Cleaned Dataset, EDA Attempts and Code Archives can be found [here](https://drive.google.com/drive/folders/1A4AH3E1vJ7tGiC_hhnfZ8XMC6pPnu8E5?usp=sharing)

### Message to Collaborator(s)

To : Poly

When I wrote the current challenges, I realized this is beyond scope of my understanding a lot. Imagine actually analyzing data from small sample size (yes, small) and trying to understand what the model yields, or even actually building statistical models from scratch. If we successfully conclude this job, we probably can even do any real world data we wish to do with our magic.

I think this will be the most challenging job we have to do so in the osu!mania scene so far. I'm looking forward to learn and discuss with you. You are really smart and I believe we can do something to contribute the osu!mania community. Thank you for joining me and have fun with your work. (and don't forget to take a break as well as it is important to you)

\- IndexError
