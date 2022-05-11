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

### Exploratory Data Analysis and Data Imputation

We transform the scores using the function below

<img src="https://latex2png.com/pngs/c43dc4aebb0ef3a4cafcb81b7b82a4ea.png" /> 

where **score** variable were normalized by dividing 1000000

This function can be seen as an inverse logit function and it (kind of) make the score data looks more normal (except for example Qualifiers Stage 4 where p-value of chi-squared goodness of fit test goes boom)

Then we use [k-Nearest Neighbors Imputation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) to impute the missing data and then using [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) as a dimensionality reduction techniques to plot each scores into 2D plane, each beatmap type and round are plotted individually. For example, this is the EDA result of Qualifiers Stage
<div align="center">
<img src="https://cdn.discordapp.com/attachments/546525809440194560/973614398683881553/Q.png" alt="PCA Results from Qualifiers Stage Scores" />
</div>

### Outlier Detection

Currently we have the various models for Outlier Detection Algorithm, we decide to use the results from Exploratory Data Analysis (with PCA n-compoenents = 3) to do the following :

**Clustering using K-means**

We use [K-means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) (k=7) to cluster the players into groups and then we plan to detect the outliers from those groups where the group member is a small minority of the population (for example a group with 1 member). This is one of the results from Qualifiers Stage

<div align="center">
<img src="https://cdn.discordapp.com/attachments/546525809440194560/973618310660911195/Q.png" alt="K-Means Qualifiers Stage" />
</div>

**One Class SVM**

We use [One-Class Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) with [RBF Kernel](https://towardsdatascience.com/svm-classifier-and-rbf-kernel-how-to-make-better-models-in-python-73bb4914af5b) to detect the outliers from the data. This is one of the results from Semifinals Stage

<div align="center">
<img src="https://cdn.discordapp.com/attachments/546525809440194560/973619336122097686/SF.png" alt="One-Class SVM" />
</div>

**Isolation Forest**

We also use [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) to detect the outliers from the data. This method relies on how easy it is to classify the data using the Unsupervised Tree. This is one of the results from Semifinals Stage

<div align="center">
<img src="https://cdn.discordapp.com/attachments/546525809440194560/973621954013716581/SF.png" alt="Isolation Forest" />
</div>

**Local Outlier Factor**

[Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) is a method to find the outlier using the distance of Nearest Neighbors (according to sklearn it is local density but yeah that is almost the same thing I guess) and we use those to calculate the anomaly score to detect the outliers. This is the example from Round of 32

<div align="center">
<img src="https://cdn.discordapp.com/attachments/546525809440194560/973623292344795196/RO32.png" alt="LOF" />
</div>

### Skillbanning

See `Problems / Challenges`

---

## Code Documentation

This Code Documentation will focus on the class `Dataset`

### Dataset Class

**Initialization**

We can initialize the class by using the `.csv` file path

```python {all|2|1-6|9|all}
from dataset import Dataset

ds = Dataset('4dm_logit.csv')
ds.data # pd.DataFrame
```

**Query**

We can query the dataset by inputting the `round` and/or `beatmap_type` into the `Dataset.query` function

```python {all|2|1-6|9|all}
ds.query(round="SF") # query Semifinals with Player Data
ds.query(beatmap_type="LN") # query LN maps with Player Data
ds.query(numeric=True) # query all without Player Data
```

**Remove All Null Player Data**

We can removing the player that has no data in the dataset by using the command `Dataset.remove_unplayers()` function

```python {all|2|1-6|9|all}
SF_dataset = ds.query(round="SF") # query Semifinals with Player Data
SF_dataset.remove_unplayers() # remove players who don't play in semifinals out
```

**Applying Models**

We can apply the models to the specified data by using `Dataset.apply_outlier_model` function

```python {all|2|1-6|9|all}
from dataset import Dataset
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

LOGIT_DATASET = Dataset("4dm_logit.csv").query(numeric=False) # logit dataset with player data

# This function returns model sequence which consists of KNN Imputer and PCA with the inputted pca_dim
def get_model(pca_dim):
    return Pipeline([('imputer', KNNImputer()), ('pca', PCA(pca_dim))])

# Use Model with KNN Imputer and PCA with n_components = 3 (3 dimensional)
pca_model = get_model(3)
# use One-Class SVM as an Outlier Detection Modek
oneclassSVM = OneClassSVM()

# Apply the dimensionality reduction and outlier models into the dataset
# This returns the players list, the result of PCA or other dimensionality reduction model and the classification result of unsupervised models
players, pca_res, outlier_res = LOGIT_DATASET.apply_outlier_model(None, "SF", pca_model, oneclassSVM)
```

---

## Problems / Challenges

### Current Obstacles

**Interpretability of the Model**

With all the models we have selected, the problem is the interpretability of the procedure to the public. We need to understand what KNN-Imputation and PCA results tell us and how the outlier models actually select the outliers (calculation and training process). As we are aspiring Data Scientists, this is a challenging job to do to communicate with people and actually find the directions of what to do next.

**Model Selection / More Model Ideas**

We need to find the metric to decide whether which model works the best for this dataset and we need more ideas on the models too (probably add some twists to old model or suggest new ones)

### Current Challenges

**Skillbanning**

After we found the outliers, we need to select the players to skillban. This process is difficult to do automatically since there is no model that supports this yet so we need to come up with our own. This also caused from the problem of **Interpretability of the Model** since we still don't understand the result of dimensionality reduction techniques clearly yet.

---

## Editor's Notes

To : Poly

When I wrote the current challenges, I realized this is beyond scope of my understanding a lot. Imagine actually analyzing data from small sample size (yes, small) and trying to understand what the model yields, or even actually building statistical models from scratch. If we successfully conclude this job, we probably can even do any real world data we wish to do with our magic.

I think this will be the most challenging job we have to do so in the osu!mania scene so far. I'm looking forward to learn and discuss with you. You are really smart and I believe we can do something to contribute the osu!mania community. Thank you for joining me and have fun with your work. (and don't forget to take a break as well as it is important to you)

\- IndexError_