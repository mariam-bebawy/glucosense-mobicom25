# %%
from sklearn import metrics
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import shap

from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
from cuml.svm import SVR as cuSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV 
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA

import os
os.chdir("/local-scratch/GlucoseProject/mobicom23_mobispectral/regression")
import sys
sys.path.append(os.getcwd())
from utils import *
from config import *

# %%
BANDS_WAVELENGTHS = [397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68, 932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58]

# %%
num_features_to_select = 50

# %%
subject_id_test = [2]
# Everything else from the subject_id_test
subject_id_train = [1, 3]

# %%
filter_rows_by_id("../" + HYPER_LABELS_FILE, "../" + TEST_FILE, subject_id_test)
filter_rows_by_id("../" + HYPER_LABELS_FILE, "../" + TRAIN_FILE, subject_id_train)

# %% [markdown]
# ## Data understanding

# %%
X, y, image_list_train, image_list_test = get_all_data('../datasets/dataset_glucose/Hyperspectral', 'train_data.csv', 'test_data.csv', 1, 'hyper')

# %%
X, y, image_list_train, image_list_test = get_all_data('../datasets/dataset_glucose/Hyperspectral', 'train_data_temp.csv', 'test_data_temp.csv', 1, 'hyper')

# %%
data_root = '../datasets/dataset_glucose/mobile/GooglePixel4XL'
file_train = 'mobi_train.csv'
file_test = 'mobi_test.csv'
step = 1

# %%
data_root = '../datasets/dataset_glucose/mobile/RaspberryPi-RGB+ToF'
file_train = 'mobi_train.csv'
file_test = 'mobi_test.csv'
step = 1

# %%
X, y, image_list_train, image_list_test = get_all_data(data_root, file_train, file_test, step, 'mobi')

# %%
low_signal = image_list_train.get_low_signal(65, 0.15)

# %%
print("Low signal images: ", low_signal)

# %%
import matplotlib.pyplot as plt

# Example data
data = [1, 2, 5, 7, 8, 8, 10, 12, 14, 15, 18, 20, 21, 22, 25]

# Creating a boxplot
plt.boxplot(data)
plt.title('Boxplot Example')
plt.ylabel('Values')
plt.show()


# %%
# Summary statistics
print("Mean:", np.mean(X, axis=0))
print("Mean size: ", np.mean(X, axis=0).shape)
print("Median:", np.median(X, axis=0))
print("Standard Deviation:", np.std(X, axis=0))

# Visualizations
plt.hist(X, bins=200)
plt.ylabel('Frequency')
plt.xlabel('signatures value')
plt.show()

# X has only 68 bands on the x-axis, BANDS_WAVELENGTHS has 204
# Every x tick is 3 apart from each other in the BANDS_WAVELENGTHS
plt.boxplot(X)
plt.xticks(range(1, 69), BANDS_WAVELENGTHS[::3], rotation=90)
plt.show()

# Correlation
correlation_matrix = np.corrcoef(X, rowvar=False)
print(correlation_matrix)
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.show()

# %%
# Summary statistics
print("Mean:", np.mean(X, axis=0))
print("Mean size: ", np.mean(X, axis=0).shape)
print("Median:", np.median(X, axis=0))
print("Standard Deviation:", np.std(X, axis=0))

# Visualizations
plt.hist(X, bins=200)
plt.ylabel('Frequency')
plt.xlabel('signatures value')
plt.show()

plt.boxplot(X)
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.show()

# Correlation
correlation_matrix = np.corrcoef(X, rowvar=False)
print(correlation_matrix)
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.show()

# %%
data_root = '../datasets/dataset_glucose/Hyperspectral'
file_train = 'train_data.csv'
file_test = 'test_data.csv'
step = 1

# %%
data_root = '../datasets/dataset_glucose/Hyperspectral'
file_train = 'train_data_temp.csv'
file_test = 'test_data_temp.csv'
step = 1

# %%
data_root = '../datasets/dataset_glucose/mobile/GooglePixel4XL'
file_train = 'mobi_train.csv'
file_test = 'mobi_test.csv'
step = 1

# %% [markdown]
# ## Feature understanding

# %%
X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, step, 'mobi')
# X_train, X_val, y_train, y_val = TTS(X_train, y_train, test_size=0.3, random_state=42)

# %%
X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, step, 'hyper')
# X_train, X_val, y_train, y_val = TTS(X_train, y_train, test_size=0.3, random_state=42)

# %%
# Scale data
# scaler = StandardScaler()
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# %%
# Scale data
# scaler = StandardScaler()
scaler = RobustScaler()
X = scaler.fit_transform(X)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)

# %%
for row in X:
    plt.plot(row)

# %%
import cupy as cp
X_train = cp.array(X_train)
# X_val = cp.array(X_val)
X_test = cp.array(X_test)

# %%
model = XGBRegressor(n_estimators=250
           ,subsample=0.5236842105263158
           ,random_state=42
           ,tree_method='hist'
           ,device='cuda'
           ,n_job=-1
           ,learning_rate=0.1)

# %%
X_train = cp.asnumpy(X_train)
# X_val = cp.asnumpy(X_val)
X_test = cp.asnumpy(X_test)

# %%
# PCC plot
correlations = []
for i in range(X.shape[1]):
    corr, _ = pearsonr(X[:, i], y)
    correlations.append(corr)

plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS)), correlations)
plt.xlabel('Wavelength (nm)')
plt.ylabel('PCC')
plt.title('PCC of selected bands')
# xticks is band wavelengths, but 10 apart from each tick
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.show()

# %%
# Calculate mutual information
# 10 Subjects 121 hyper images
mi = mutual_info_regression(X, y)

# Plot mutual information
plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS)), mi)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Mutual Information')
plt.title('Mutual Information of selected bands')
# xticks is band wavelengths, but 10 apart from each tick
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.show()

# %%
# Calculate mutual information
# 11 subjects 157 hyper images
mi = mutual_info_regression(X, y)

# Plot mutual information
plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS)), mi)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Mutual Information')
plt.title('Mutual Information of selected bands')
# xticks is band wavelengths, but 10 apart from each tick
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.show()

# %%
# Robust Scaler, without x_val, Step 1
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
X_new = selector.fit_transform(X_train, y_train)

selected_feature_indices = selector.get_support(indices=True)

print('Selected bands: ', selected_feature_indices)

plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS)), selector.scores_)
plt.xlabel('Wavelength (nm)')
plt.ylabel('F value')
plt.title('F value of selected bands')
# xticks is band wavelengths, but 25 apart from each tick
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.show()

# %%
# Robust Scaler, without x_val
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
X_new = selector.fit_transform(X_train, y_train)

selected_feature_indices = selector.get_support(indices=True)

print('Selected bands: ', selected_feature_indices)

plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS)), selector.scores_)
plt.xlabel('Feature')
plt.ylabel('F value')
plt.title('F value of selected bands')
# xticks is band wavelengths, but 25 apart from each tick
plt.xticks(range(0, 204, 25), BANDS_WAVELENGTHS[0::25], rotation=90)
plt.show()

# %%
# Robust Scaler, without x_val, Step 1, Full dataset
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
X_new = selector.fit_transform(X, y)

selected_feature_indices = selector.get_support(indices=True)

print('Selected bands: ', selected_feature_indices)

plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS)), selector.scores_)
plt.xlabel('Wavelength (nm)')
plt.ylabel('F value')
plt.title('F value of selected bands')
# xticks is band wavelengths, but 25 apart from each tick
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.show()

# %%
# Robust Scaler, without x_val, Step 2
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
X_new = selector.fit_transform(X, y)

selected_feature_indices = selector.get_support(indices=True)

print('Selected bands: ', selected_feature_indices)

plt.figure(figsize=(20, 10))
plt.bar(range(0, len(BANDS_WAVELENGTHS), 2), selector.scores_)
plt.xlabel('Wavelength (nm)')
plt.ylabel('F value')
plt.title('F value of selected bands')
# xticks is band wavelengths, but 10 apart from each tick
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.show()

# %%
# Robust Scaler, without x_val, Step 3
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
X_new = selector.fit_transform(X, y)

selected_feature_indices = selector.get_support(indices=True)

print('Selected bands: ', selected_feature_indices)

plt.figure(figsize=(20, 10))
plt.bar(range(0, len(BANDS_WAVELENGTHS), 3), selector.scores_)
plt.xlabel('Wavelength (nm)')
plt.ylabel('F value')
plt.title('F value of selected bands')
# xticks is band wavelengths, but 10 apart from each tick
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.show()

# %%
# Robust Scaler, without x_val, Step 5
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
X_new = selector.fit_transform(X, y)

selected_feature_indices = selector.get_support(indices=True)

print('Selected bands: ', selected_feature_indices)

plt.figure(figsize=(20, 10))
plt.bar(range(0, len(BANDS_WAVELENGTHS), 5), selector.scores_)
plt.xlabel('Wavelength (nm)')
plt.ylabel('F value')
plt.title('F value of selected bands')
# xticks is band wavelengths, but 10 apart from each tick
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.show()

# %%

# total_features = X_train.shape[1]
# rfe = RFE(model, n_features_to_select=num_features_to_select)

# # Fit the RFE to the training data
# rfe.fit(X_train, y_train)

# ranking = rfe.ranking_

# %%
# ranking

# %%
total_features = X_train.shape[1]
num_irrelevant_features_eliminated = np.arange(total_features - 1, -1, -1)
print(num_irrelevant_features_eliminated)

# %%
model = XGBRegressor(
        random_state=42
        ,tree_method='hist'
        ,device='cuda'
        ,n_job=-1
        )

# %%
model = cuRF(random_state=42)
model = RandomForestRegressor(random_state=42)

# %%
model = MLPRegressor(random_state=42)

# %%
# Support vector regression
model = cuSVR(kernel='rbf')

# %%
# KNN
model = cuKNeighborsRegressor()

# %%
# Shap values
shap_model = model
shap_model.fit(X_train, y_train)

# %%
# Create a function to generate predictions (required by SHAP)
def predict_fn(X):
    return shap_model.predict(X)


# %%
# Create a SHAP explainer
explainer = shap.Explainer(shap_model, algorithm='auto')
# Compute SHAP values for the test set
shap_values = explainer(X_test)


# %%
# Create a SHAP explainer
explainer = shap.TreeExplainer(shap_model)
# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# %%
# Shap values
# SVR, KNN, MLP
X_sample = shap.sample(X_train, 3000)
# Create a SHAP explainer
explainer = shap.KernelExplainer(predict_fn, X_sample)
# Compute SHAP values for the test set
sample = shap.sample(X_test, 1000)
shap_values = explainer.shap_values(sample)


# %%
# Shap values
# neural networks
shap_model = model
shap_model.fit(X_train, y_train)

# Create a SHAP explainer
explainer = shap.DeepExplainer(predict_fn, X_train)
# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)


# %%
# XGBoost
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=50)

# %%
# XGBoost
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=50)

# %%
# Calculate mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

# Plot using matplotlib
plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS)), mean_abs_shap_values)
plt.ylabel('Mean |SHAP Value|')
plt.xlabel('Feature')
plt.title('Feature Importance based on Mean Absolute SHAP Values')
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)

# # Save the plot to a file
# plt.savefig('XGB_shap_summary_plot.png', bbox_inches='tight')
plt.show()

# %%
# XGBoost
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=50, plot_type='bar')

# %%
# Random Forest
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=50)

# %%
# Random Forest
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=50, plot_type='bar')

# %%
# SVR
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=40, plot_type='bar')

# %%
# MLP
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=40)

# %%
# MLP
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=40, plot_type='bar')

# %%
# KNN
# The summary plot gives a global overview of the feature importance
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=50)

# %%
# KNN
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=40, plot_type='bar')

# %%
print(len(shap_values))

# %%
# The dependence plot shows the relationship between the SHAP value of a single feature 
# and the feature value itself, highlighting interactions with other features.
shap.initjs()
shap.dependence_plot("rank(0)", shap_values.values, X_test, feature_names=BANDS_WAVELENGTHS, interaction_index=None)
# interaction_index“auto”, None, int, or string
# The index of the feature used to color the plot. The name of a feature can also be passed as a string. If “auto” then shap.common.approximate_interactions
# is used to pick what seems to be the strongest interaction (note that to find to true stongest interaction you need to compute the SHAP interaction values).


# %%
# The force plot provides a detailed explanation for a single prediction, 
# showing how each feature contributes to pushing the prediction from the base value to the final prediction.
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values.values[0, :], X_test[0, :], feature_names=BANDS_WAVELENGTHS)

# %%
# Random Forest
# The summary plot gives a global overview of the feature importance 
# and the effects of the features on the model's predictions.
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=40)

# %%
accuracy_list = []
for i in num_irrelevant_features_eliminated:
    rfe = RFE(model, n_features_to_select=(total_features - i))
    rfe.fit(X_train, y_train)
    y_pred = rfe.predict(X_test)
    accuracy = MSE(y_test, y_pred)
    accuracy_list.append(accuracy)

plt.figure(figsize=(8, 6))
plt.plot(num_irrelevant_features_eliminated, accuracy_list, marker='o')
plt.xlabel("Number of Irrelevant Features Eliminated")
plt.ylabel("Accuracy")
plt.title("Model Performance as Irrelevant Features are Eliminated (RFE)")
plt.grid(True)
plt.show()

# %% [markdown]
# ## PCA

# %%
PCA_model = PCA()
PCA_model.fit(X_train)
# visualize the selelcted components
plt.plot(np.cumsum(PCA_model.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# %% [markdown]
# ## Random Forest

# %%
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# Visualize feature importance
plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS)), rf_model.feature_importances_)
plt.xlabel("Feature index")
plt.ylabel("Feature importance")
plt.xticks(range(0, 204, 25), BANDS_WAVELENGTHS[0::25], rotation=90)
plt.title("Feature importance of all features")
plt.show()

# Select top features
top_idx = np.argsort(rf_model.feature_importances_)[::-1][:num_features_to_select]
print(f"Top {num_features_to_select} features: {top_idx}")

# %%
# top_idx is the index in the BANDS_WAVELENGTHS array
# I want to the corresponding content stored in BANDS_WAVELENGTHS array
wavelengths = [BANDS_WAVELENGTHS[i] for i in top_idx]
print(wavelengths)

# %%
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X, y)

# Visualize feature importance
plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS)), rf_model.feature_importances_)
plt.xlabel("Feature index")
plt.ylabel("Feature importance")
plt.xticks(range(0, 204, 25), BANDS_WAVELENGTHS[0::25], rotation=90)
plt.title("Feature importance of all features")
plt.show()

# Select top features
top_idx = np.argsort(rf_model.feature_importances_)[::-1][:num_features_to_select]
print(f"Top {num_features_to_select} features: {top_idx}")

# %% [markdown]
# ## XGBoost

# %%
# Robust Scaler, without x_val
xgb_model = XGBRegressor(device='cuda', tree_method='hist', random_state=42, n_jobs=-1, objective='reg:squarederror')
xgb_model.fit(X, y)

# Visualize feature importance
plt.figure(figsize=(10, 5))
plt.bar(range(0,len(BANDS_WAVELENGTHS),1), xgb_model.feature_importances_)
plt.xlabel("Feature index")
plt.ylabel("Feature importance")
plt.xticks(range(0, 204, 25), BANDS_WAVELENGTHS[0::25], rotation=90)
plt.title("Feature importance of all features")
plt.show()

# Select top features
top_idx = np.argsort(xgb_model.feature_importances_)[::-1][:num_features_to_select]
print(f"Top {num_features_to_select} features: {top_idx}")

# %%
# top_idx is the index in the BANDS_WAVELENGTHS array
# I want to the corresponding content stored in BANDS_WAVELENGTHS array
wavelengths = [BANDS_WAVELENGTHS[i] for i in top_idx]
print(wavelengths)

# %%
# Standard scaler, without x_val
xgb_model = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# # Visualize feature importance
# plt.figure(figsize=(10, 5))
# plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
# plt.xlabel("Feature index")
# plt.ylabel("Feature importance")
# plt.title("Feature importance of all features")
# plt.show()

# # Select top 50 features
# top_50_idx = np.argsort(xgb_model.feature_importances_)[::-1][:50]
# print("Top 50 features: ", top_50_idx)

# Visualize feature importance
plt.figure(figsize=(10, 5))
plt.bar(range(0,len(BANDS_WAVELENGTHS),1), xgb_model.feature_importances_)
plt.xlabel("Feature index")
plt.ylabel("Feature importance")
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
plt.title("Feature importance of all features")
plt.show()

# Select top features
top_idx = np.argsort(xgb_model.feature_importances_)[::-1][:num_features_to_select]
print(f"Top {num_features_to_select} features: {top_idx}")

# %% [markdown]
# ## Lasso Regression

# %%
# Fit LassoCV model 
lasso_cv = LassoCV(cv=10, random_state=42, verbose=True, alphas=[0.0001, 0.001, 0.01, 0.1, 1, 5], n_jobs=-1)
lasso_cv.fit(X, y)

print("LassoCV model fit complete...")
# Feature selection 
sfm = SelectFromModel(lasso_cv, prefit=True, threshold=-np.inf, max_features=num_features_to_select) 
# X_train = sfm.transform(X) 
# X_val = sfm.transform(X_val)
# X_test = sfm.transform(X_test)

# Show dimensions of data after LassoCV
# print("X_train_selected shape: ", X_train.shape)
# print("X_val_selected shape: ", X_val.shape)
# print("X_test_selected shape: ", X_test.shape)

feature_indices = np.array(range(len(lasso_cv.coef_)))

# Analyze selected features and their importance 
selected_feature_indices = np.where(sfm.get_support())[0] 
selected_features = feature_indices[selected_feature_indices] 
coefficients = lasso_cv.coef_ 

# Visualize selected features and their importance
plt.figure(figsize=(10, 5))
plt.bar(feature_indices, coefficients)
plt.xticks(selected_features)
plt.xlabel("Feature index")
plt.ylabel("Feature importance")
plt.title("Feature importance of selected features")
plt.show()

# Show selected features
print("Selected features: ", selected_feature_indices)


# %%
# Fit LassoCV model 
lasso_cv = LassoCV(cv=10, random_state=42, verbose=True, alphas=[0.0001, 0.001, 0.01, 0.1, 1, 5], n_jobs=-1)
lasso_cv.fit(X_train, y_train)

print("LassoCV model fit complete...")
# Feature selection 
sfm = SelectFromModel(lasso_cv, prefit=True, threshold=-np.inf, max_features=num_features_to_select) 
X_train = sfm.transform(X_train) 
X_val = sfm.transform(X_val)
X_test = sfm.transform(X_test)

# Show dimensions of data after LassoCV
print("X_train_selected shape: ", X_train.shape)
print("X_val_selected shape: ", X_val.shape)
print("X_test_selected shape: ", X_test.shape)

feature_indices = np.array(range(len(lasso_cv.coef_)))

# Analyze selected features and their importance 
selected_feature_indices = np.where(sfm.get_support())[0] 
selected_features = feature_indices[selected_feature_indices] 
coefficients = lasso_cv.coef_ 

# Visualize selected features and their importance
plt.figure(figsize=(10, 5))
plt.bar(feature_indices, coefficients)
plt.xticks(selected_features)
plt.xlabel("Feature index")
plt.ylabel("Feature importance")
plt.title("Feature importance of selected features")
plt.show()

# Show selected features
print("Selected features: ", selected_feature_indices)

# %%
# Show image
file = '../datasets/dataset_glucose/Hyperspectral/HS_GT/2749.mat'

bands = 0
show_image(file, bands)


