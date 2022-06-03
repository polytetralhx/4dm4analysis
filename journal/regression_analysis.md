# Regression Analysis Documentation

Some say that the physical ability of an osu!mania player to play tougher charts may translate into stronger accuracy and consistency. Hence, there could be a strong correlation between physicality and accuracy of a player (with a small number of exceptions).

It may not make sense to analyse both types of players as if they are 2 different forms of life. _Wouldn't it be better to marry both aspects and study how they affect a player's tournament performance?_

This section of the data analysis aims to document the scores obtained by a player over the rounds they played while in 4DM4, and aims to find answers to a couple of questions we are also curious about:

- Can the average 4 digit player have their scores over each of the rounds modelled by a mathematical equation?
- If so, what kind of equation should the player's scores be modelled to?
- If so, can we also model an outlier 4 digit by an equation that can be statistically proven to differ from the equation for the average 4 digit? Can we detect outliers through this statistical approach?

## Preparation of Data

Huge thanks to HowToPlayLN (HowToProgramming) for contributing much of the code and engineering processes! Refer to the [readme documentation](readme.md) for more details regarding the dataset and class Dataset used for this project.

## Statistical Analysis Methods

### Linear Regression

Linear Regression is one of the basic approaches to model the relationship between a set of dependent and independent variables, **under the assumption that it is linear**. For this subproject, the average scores for each player in each of the categories (RC, LN, HB) were computed and plotted against each round in the tournament (excluding Qualifiers).

The [Linear Regression model from the sklearn package](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) was applied to these scores to identify the best-fit line (i.e. the linear equation to model how an average participant's scores will change over the rounds).

### Polynomial Regression

Polynomial Regression is the other form of regression considered for this sub-problem. This is preferred over linear regression in the event where a linear equation is unable to capture the patterns in score changes over the rounds.

Of course, when it comes down to a polynomial, there is an implication that a player's scores could be related to the difficulty of a round through a quadratic equation, or a cubic equation. The difference in powers is called a _degree_, where a quadratic equation would be of the _2nd degree_, and a cubic equation of the _3rd degree_ (basic mathematical knowledge is expected).

For this specific project, the _2nd degree_ and _3rd degree_ best-fit curve was identified for the same scores. The Polynomial Regression model from the sklearn package has also been used.

### Shrinkage Methods for Regression

In the event that scores of each player share multiple relationships with one another and thus cannot have a direct relationship between 2 variables (i.e. predictor variables in the data experience _multicolinearity_), Ridge Regression is a model tuning method that attempts to reduce error by adding a passive bias to the estimates.

On the other hand, the Lasso Regression model was also applied. This model offers more predictive accuracy (i.e. the prediction will be closer to the true model), by reducing some coefficients of the final model(sometimes to 0). This is better used when there are less observations and more features in the data.

https://corporatefinanceinstitute.com/resources/knowledge/other/ridge/
https://corporatefinanceinstitute.com/resources/knowledge/other/lasso/

### In-Depth Analysis on Ridge and Lasso

The difference between Ridge and Lasso can be found in the optimization process. A constrained optimization problem of Ridge can be written as

![Optimization Problem of Ridge Regression](https://cdn.discordapp.com/attachments/546525809440194560/982146458474119168/unknown.png)

<div align="center">Optimization Problem of Ridge Regression (<a href="https://doi.org/10.1007/978-0-387-84858-7_8">The elements of statistical learning</a>)</div>

Which can be written as an unconstrained optimization using the [**Lagrange Multipiler**](https://en.wikipedia.org/wiki/Lagrange_multiplier) which is the method to include the constraint into an equation that is required to be optimized.

![Optimization Problem of Ridge Regression with Lagrange Multipiler](https://cdn.discordapp.com/attachments/546525809440194560/982147696267784242/unknown.png)

<div align="center">Unconstrained Optimization Problem of Ridge Regression (<a href="https://doi.org/10.1007/978-0-387-84858-7_8">The elements of statistical learning</a>)</div>

This Unconstrained Optimization Problem can be solved analytically. The analytical solution of Multivariable Ridge Regression can be written as

![](https://cdn.discordapp.com/attachments/546525809440194560/982148685716668466/unknown.png)

<div align="center">Analytical Solution of Ridge Regression (<a href="https://doi.org/10.1007/978-0-387-84858-7_8">The elements of statistical learning</a>)</div>

We can see that Ridge tries to reduce the variance from outliers by sacrificing the bias according to constraints meanwhile also solving the multicollinearity in multivariate regression analysis.

Lasso also solves the same problem with the different approach, here the constraints are the absolute value of the coefficients.

![](https://cdn.discordapp.com/attachments/546525809440194560/982155091433885696/unknown.png)

<div align="center">Constrained Optimization Problem of Lasso (<a href="https://doi.org/10.1007/978-0-387-84858-7_8">The elements of statistical learning</a>)</div>

Which can be written as an unconstrained optimization problem using Lagrange Multipiler

![](https://cdn.discordapp.com/attachments/546525809440194560/982155167455657984/unknown.png)

<div align="center">Unconstrained Optimization Problem of Lasso (<a href="https://doi.org/10.1007/978-0-387-84858-7_8">The elements of statistical learning</a>)</div>

Due to the non-differentiability of the function. We cannot find the analytical solution of this optimization problem. However we can see that Ridge and Lasso serves the same purpose of reducing the variance from the outliers with the bias tradeoff. However according to [this section](#shrinkage-methods-for-regression) we can see that Lasso tends to ignore some of the unrelevant variables and outliers.

## Interpretations and Evaluations

At a glance, all models generated by each of the 4 regression methods possess a _leading coefficient_ negative in value. This implies that all models agree that scores of the players decrease over the rounds. However, how these scores are predicted to decrease differ for each model. This lends for varying predictive accuracies, which are evaluated to see if the model can truly represent how players' scores change over the rounds.

Of all the models, the linear regression models are considered better models for the player scores. This is done through by running the **one-way analysis of variance (ANOVA) test** and calculating the **p-value** of these models.

### F-test for Regression Analysis

**F-test for Regression Analysis** or **one-way analysis of variance (ANOVA) test** is used to conduct the test whether there is a linear relationship between an independent variable and a dependent variable using **Sum of Squares due to Regression** and **Sum Squared Error** using the following formulas

![](https://cdn.discordapp.com/attachments/546525809440194560/982164637426520095/unknown.png)

We then create the null and alternative hypotheses

![](https://cdn.discordapp.com/attachments/546525809440194560/982166044330303508/unknown.png)

and obtain the f-statistics using the following formula

![](https://cdn.discordapp.com/attachments/546525809440194560/982167463162695760/unknown.png)

Note that **Sum of Squares due to Regression** has the degree of freedom 1 and **Sum Squared Error** has the degree of freedom n-2 (Detailed Explanation can be found [here](https://math.stackexchange.com/questions/626732/linear-regression-degrees-of-freedom-of-sst-ssr-and-rss) tbh I am still spinning my head over this)

Then we conduct the Right tail test for f-statistics, the p-value of the test can be found by using the formula

![](https://cdn.discordapp.com/attachments/546525809440194560/982168192640225280/unknown.png)

As it is a right-tailed test, a higher **F-statistics** implies that the tested model, in layman terms, fits better with the data. A **p-value**, if lower than the defined **level of significance**, also means that the relationship proposed by the regression model is more likely to be valid.

In other words, **it can be confirmed that the the 4-digit player's average logit score over each round and every skillset follow a negative and linear relationship.**

### In-depth Analysis of one-way ANOVA

To recap, **One-way analysis of variance (ANOVA) test** used **f-test** to perform the test for linear relationship using **Sum of Squares due to Regression** and **Sum Squared Error**.

Normally, **f-test** is used to conduct the test for hypothesis of ratio of variances (i.e. variance comparison). **Right-tailed f-test** hypotheses can be written as

- Null Hypothesis : The variance from sample 1 is **less than or equal to** the variance from sample 2
- Alternative Hypothesis : The variance from sample 1 is **greater than** the variance from sample 2

Extrapolating further into the distribution of ratio of variances which sample size has the degree of freedom `df1` and `df2`. We use the below formula to obtain the **f-statistics**

![](https://cdn.discordapp.com/attachments/546525809440194560/982182317722320926/unknown.png)

and perform the right-tail test with degree of freedom `df1` and `df2`

We then noticed that `s1^2` is basically **Sum of Squares due to Regression (SSR)** or according to degree of freedom `df1 = 1`, it is the **Mean of Squares due to Regression (MSR)** and `s2^2` is **Sum of Squares Error (SSE) divided by n-2** which is **Mean of Squares Error (MSE)** according to degree of freedom `df2 = n-2`.

Basically, the **One-way ANOVA** tries to test whether **Mean of Squares due to Regression** is greater than **Mean of Squares Error** and this implies the **linear dependency** of two variables. To illustrate, imagine a regression line with zero slope and intercept is a mean value of our dependent variable marked as `l1` and our regression line which deprends on our independent variable as `l2`. The **Mean of Squares due to Regression** is the **Mean of Squares Error** of `l1` and **Mean of Squares Error** of `l2` is **Mean of Squares Error**. To determine if `l2` is a better fit than `l1` (in other words, there is a significant linear relationship between two variables), you need to conduct a test whether **Mean of Squares Error** is less than **Mean of Squares due to Regression**, in other words, **Mean of Squares due to Regression** is greater than **Mean of Squares Error**. Which we need to use **f-test** in this analysis.

### Interpretation of p-value

**p-value** is a probability that the observed value is true given the hypothesis is true or `Pr(Observation | H0)`, and **not to be confused with** `Pr(H0 | Observation)`, they are not the same thing, in fact

![](https://cdn.discordapp.com/attachments/546525809440194560/982189938428764220/unknown.png)

but if the p-value (`Pr(Observation | H0)`) is low enough everything is gonna be okay trust me.

**p-value** indicates how likely the **Alternative Hypothesis** will be accepted. The chance that Alternative Hypothesis will be accepted is approximately `1 - p`. So if the **p-value** is lower than the **significance level** (mostly `0.05`), we then **Reject the Null Hypothesis** as a wise man said hundred years ago, and **Accept the Alternative Hypothesis**.

### Shapiro-Wilk test for Residual Analysis

<br>
<div align="center"><em>"There is no turning back when you commit a Normal Distribution."</em></div>

<br>

**Shapiro-Wilk test** is used to test whether the given dataset is **Normally Distributed** by conducting two hypotheses

- Null Hypothesis : The dataset is Normally Distributed
- Alternative Hypothesis : The dataset is not Normally Distributed

More details can be found in [Wikipedia](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test) or [This Document by MIT](https://math.mit.edu/~rmd/465/shapiro.pdf). It is really complicated to decompose this. (tbh I still don't understand this completely)

Our implementation of **Shapiro-Wilk test** is to check if the residual from regression is **Normally Distributed** which is [one of the assumptions in Linear Regression](https://analyse-it.com/docs/user-guide/fit-model/linear/residual-normality#:~:text=Normality%20is%20the%20assumption%20that,Shapiro%2DWilk%20or%20similar%20test.)

From the tests, we obtained the p-value from Linear and Polynomial Regression of each beatmap category using [scipy.stats.shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html) function. The result is the `RC` and `LN` has the non-normality of residuals in all of linear and polynomial models due to the small p-value, meanwhile `HB` category p-value is approximately 0.5 in Linear Regression Model which can be interpreted that Linear Regression Model is valid in `HB` category but not other categories.

## Limitations

There are of course, some assumptions made in the model which may affect the result of this test:

- Mappool difficulty scaling can definitely play a part in the players' performance over each round. Hence, making use of a linear scale for each round may not be entirely accurate. There is a possibility that the established linear relationship is not ground on decent premises -- Mappool difficulty scaling will have to be corrected (somehow) before a second test on the same models.

## Possible Extensions

Going forward, it may then be possible to provide a justifiable response to some other questions:

- Can an outlier 4-digit have their tournament performance be different from that of the average 4-digit?
- Is it then possible to model the performances of specialist players, and differentiate them from these outliers?
- Can this linear relationship also be representative of the larger osu!mania playerbase? In other words, _can this model be scaled up/down for use by other rank-restricted tournaments?_
