# IndexError's Journal on Analysis of 4dm4 Scores

#### (Not so) Important Notes

- I prefer referring myself as _index_ in here but you can call me _htpln_ or _howtoplayln_ if you know me in game.
- ~~index can't do citation and references properly don't mind him~~
- TBA

## Previous Attempt(s)

### Methodology

Read readme.md in main branch I'll import to here soon I guess

### Why are they failed ?

TBA

### How close are we to our goal ?

TBA

## Exploratory Data Analysis

The first and the most important process in the Analysis is Exploratory Data Analysis (EDA). We tried various methods to come up with the data details and skillbanning ideas. One of EDA we came up with is 95% Confidence Interval Plotting.

### 95% Confidence Interval Plotting

#### What is Confidence Interval ?

Confidence Interval is the possible range of Population Mean according to Sample Mean. Difficulty in computing the Population Mean is the size of population can be enormous and we are nowhere near being extroverted enough to ask everyone to play the maps ([IndexError trying to shameless advertise his article, 2022](https://medium.com/@indexerror_/how-i-select-my-best-coffee-shop-hypothesis-testing-for-complete-beginners-deedaeda727e)).

#### How do we obtain the Confidence Interval ?

We obtain the Confidence Interval by taking the sample mean and sample standard deviation into the process, we use sample standard deviation to calculate standard error and multiply with the student's t-distribution inverse CDF right tail. We can write the process of calculating in this pseudocode

```
mean = mean(data)
sample_std = std(data, ddof=1)
n = len(data)
degree_of_freedom = n - 1
standard_error = sample_std / sqrt(n)
Confidence Interval = mean (+-) t_inv_cdf_rt(alpha / 2, degree_of_freedom) * standard_error
```

as a method to obtain `100(1-alpha)%` Confidence Interval

#### Confidence Interval Plotting

We use the method mentioned above to obtain a 95% Confidence Interval for Rice / Hybrid and LN maps for each round and plot it in matplotlib. This is the result from 18 Grand Finalists in 4dm4 (The distribution of scores is pretty skewed but the sample means should be normal according to CLT)

<div align="center">
<img src="https://cdn.discordapp.com/attachments/548279055477374982/979039736482103307/something.png" alt="Grand Finalists CI" width=640 height=480/>
</div>

and This is the result from all players

<div align="center">
<img src="https://cdn.discordapp.com/attachments/548279055477374982/979041859601039360/unknown.png" alt="All players CI" width=640 height=480 />
</div>

From the observation, we can see that the Average and 95% Confidence Rice Score tends to decrease consistently the further the rounds and Hybrid and LN has the bump at the transition to Mid-Late game rounds (Quarterfinals and Semifinals).

## Logit Normalization

TBA

## Missing Data Validation

### Collaborative Filtering

One way to validate missing scores or unplayed scores is **Collaborative Filtering** ([Evening, 2022](https://github.com/Eve-ning/opal/blob/master/journal/opal/out/main.pdf)), which is used to predict the rating (unplayed scores) from the similar players who has played the beatmap. There are two factors involved into this model, **Similarity** and **Existed Scores**.

LaTeX in Github won't work so I'll leave some references for you [here](https://fardapaper.ir/mohavaha/uploads/2017/11/Combining-User-Based-and-Item-Based-Collaborative-Filtering-Using-Machine-Learning.pdf).

#### Similarity Measurement

In order to obtain the Similarity, we use **Pearson's Correlation Coefficient** which acts like a **Cosine Similarity** to calculate how similar of two data points are similar to each other.

#### Adjusted CF

In order to scale the data with more similarity evidences, We multiply `len(a and b not null) / len(a)` to the Similarity Measurement and got the **Adjusted Similarity Measurement** which is used to do collaborative filtering.

#### Penalization

Practically the scores of players who don't play the maps are lower than the players who plays the map. We need to add the penalty to the neighborhood we obtained from CF method. This is still on-hold and I have no idea how to add that.

### KNN Imputation

KNN Imputation is a built-in method in [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html). Which implements the K-Nearest Neighbors method into validating the missing data using the non-null Euclidean Distance.

#### K-Nearest Neighbors

K-Nearest Neighbors (KNN) is the method to predict the unknown value by using the values of the closest data points. The closest data points are mostly determined using [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance) and after obtaining the closest data points, average the prediction values of each data point.

#### How it helps with validating data

#### Our Methodology

## Hypothesis Testing

### Hypotheses

### Methodology : t-test and f-test

The Author is lacking of Bayesian stats knowledge so he uses t-test and f-test kappa

### Hypothesis Testing Results

I hate my life
