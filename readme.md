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

<br>

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

#### get_label

HowToPLayLN is bad at naming things. However this method will output the `pd.DataFrame` which consists of columns `player_name`, `avg_score` or `avg_score_logit` and `round_ord` which is the ordinal data from according to the parameter `rounds` which is a `list` of interested rounds

**Parameters**

- **rounds** : Interested Rounds, the output will be ordered according to the given list of interested rounds
- **beatmap_type** : Interested Beatmap Type, here we can input only 1 beatmap type that we are interested
- **logit** : Parameter to indicate whether the function will output `avg_score` or `avg_score_logit`

**Example**

```python {all|2|1-6|9|all}
from dataset import Dataset

_4dm4 = Dataset('4dm4.db')

interested_rounds = ["RO32", "RO16", "QF"]
interested_beatmap_type = "RC"

# Get scores with regression labeled data from the 4dm4.db sqlite dataset
# Where the interested rounds are RO32, RO16 and QF
_4dm_regression_labeled_data = _4dm4.get_label(interested_rounds, interested_beatmap_type, logit=False)

# Here rounds are encoded as the following
"""
round round_ord
RO32  0
RO16  1
QF    2
"""
_4dm_regression_labeled_data # expect pd.DataFrame with player_name, avg_score and round_ord columns
```

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

### Models implemented from sklearn

#### Polynomial Regression with Ridge and Lasso

In `models.py` there are three new models implemented from **sklearn** in order to perform the **Polynomial Regression, Polynomial Regression with Ridge and Lasso implementation** using the `sklearn.preprocessing.PolynomialFeatures` and build pipeline using `sklearn.pipeline`.

**Polynomial Regression**

We can build a **Polynomial Regression** model using

```python
import numpy as np
from models import PolynomialRegression

# Define the variables to regression
dummy_x = np.array([[1], [2], [3], [4]])
dummy_y = np.array([4.1, 8.95, 16.05, 25])

# Build a Polynomial Regression model with degree = 2
poly = PolynomialRegression(degree=2)

# Fit the Polynomial Regression model using dummies
poly.fit(dummy_x, dummy_y)

# Predict the outcome of given value(s)
poly.predict(dummy_x)
```

**Polynomial Regression with Ridge and Lasso**

There are **Ridge** and **Lasso** implementation of Polynomial Regression for conducting the study on the behaviour of the functions where they prioritize outliers less than other data points. We can build them using

```python
import numpy as np
from models import PolyRidge, PolyLasso

# Define the variables to regression
dummy_x = np.array([[1], [2], [3], [4]])
dummy_y = np.array([4.1, 8.95, 16.05, 25])

# Build a Polynomial Regression with Ridge and Lasso implementation with degree = 2 and alpha = 0.5
polyridge_alpha_0_5 = PolyRidge(degree=2, alpha=0.5)
polylasso_alpha_0_5 = PolyLasso(degree=2, alpha=0.5)

# Fit the Polynomial Regression with Ridge and Lasso implementation using dummies
polyridge_alpha_0_5.fit(dummy_x, dummy_y)
polylasso_alpha_0_5.fit(dummy_x, dummy_y)

# Predict the outcome of given value(s)
polyridge_alpha_0_5.fit(dummy_x)
polylasso_alpha_0_5.fit(dummy_x)
```

---

## Database Documentation

The actual 4dm4 data (both players and teams) are collected in `4dm4.db` which can be download through [this Google Drive Link](https://drive.google.com/file/d/12v4LsaVoniUpYK8UYq9MV-InYnXOkzJZ/view?usp=sharing). There are 4 tables in this Database.

### scores

`scores` table contains the score of each player. These are attributes of the table

- `player_name` Username of a player
- `round` A round that the score is recorded
- `beatmap_type` Type / Category of a beatmap
- `beatmap_tag` Tag of a beatmap (this attribute is used to distinguish between each map in the same beatmap category)
- `score` score of a player
- `score_logit` logit of score of a player 

### player_data

`player_data` table contains the player data and country code

- `player_name` Username of a player
- `player_id` User id of a player
- `country_code` Country Code of a player

### team_data

`team_data` table contains team data

- `country_name` Country Name of the team
- `country_code` Country Code of the team
- `last_round` The placement where the team last survived

### team_scores

`team_scores` table contains scores of each team

- `country_name` Country Name of the team
- `round` A round that the score is recorded
- `beatmap_type` Type / Category of a beatmap
- `beatmap_tag` Tag of a beatmap (this attribute is used to distinguish between each map in the same beatmap category)
- `score` score of the team recorded in the matches
- `score_logit` logit of score of the team recorded in the matches

---

## Current Problems / Challenges

TBA

---

## Editor's Notes

### Dataset, Visualizations and Code Archives (Failed Attempts that are not included in documentation)

Cleaned Dataset, EDA Attempts and Code Archives can be found [here](https://drive.google.com/drive/folders/1A4AH3E1vJ7tGiC_hhnfZ8XMC6pPnu8E5?usp=sharing)

### Message to Collaborator(s)

To : Poly

I didn't think that we will get a lot of information from this analysis. Having you here helped me a lot in this study. You are really good at storytelling those numbers. I also hope that this project will expand to not only skillbanning, but will answer some of important tournament mappool questions, whether the mappool direction, quality and difficulty balancing of the mappools and many more.

inb4 we have osu!mania Data Analysis Team lol

Again, thank you for joining me. I am looking forward for the masterpiece and I am looking forward to see you go beyond this point.

\- Index
