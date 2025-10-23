# Feature Engineering/Selection
a. First, find the correlation between each feature and the corresponding performance  

b. (bayesian learning?) Rank and select the most representative, important features

c. Lasso feature selection (Possible)


# Possible Regression Models
## 1. Gradient Boosting/XGBoost
### remove variance but not adding bias
a. if the model has low variance, gradient boosting would not help much

b. Objective is regression, we use gridsearch to find the optimal parameters
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

## 2. Support Vector Machine
The advantages of support vector machines are:

a. Effective in high dimensional spaces.

b. Still effective in cases where number of dimensions is greater than the number of samples.

c. Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

d. Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

a. If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

b. SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation 

## 3. Linear Regression
a. One of the simplest regression model
## 4. Ridge Regression
a. Advanced version of linear regression but aim to reduce the model complexity (Prevent overfit)
<br>
b. Loss function = OLS + L2 norm penalty
## 5. Lasso Regression
a. Advanced  version of linear regression but aim to reduce the model complexity (Prevent overfit)
<br>
b. Coordinate descent
## 6. Elastic-Net
a. A combined and balanced version of both the ridge and lasso regression
## 7. Bayesian Regression

## 8. Neutral Network Regression

## 9. Decision Tree Regression

## 10. Random Forest (Bagging)
### Remove bias but not adding variance (Also remove partial variance, since average over all the tree)

## 11. KNN Model
a. One of the simplest regression model

## 12. Gausian Regression

## 13. Polynomial Regression

## BERT??

## CNN


# Evaluation Approach
## ClarkeErrorGrid
a. Reference from Noninvasive Glucose Monitoring Using Polarized Light

b. a visualization of error between the predicted and ground truth
## Evaluation metric
a. Percentage of predicted glucose level which fall in Zone A or B

b. The Pearson correlation coefficients between the predicted and reference glucose concentration

c. Absolute relative differences (ARD) (A predicted glucose concentration is considered as clinical accurate
if the ARD is less 20%)

# Experiment
a. Determine the number of bands we are using, ie. [1, 204, 3] -> 68, (list the step from 1-10)

b. Determine which features are the most important, select a set of them
