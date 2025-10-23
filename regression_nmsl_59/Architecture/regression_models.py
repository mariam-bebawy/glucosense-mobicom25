# %% [markdown]
# # Regression Models with cuML, scikit-learn, and XGBoost
# This notebook demonstrates how to build and evaluate various regression models using cuML for GPU acceleration, scikit-learn for Partial Least Squares Regression, and XGBoost.

# %%
import copy
import numpy as np
import cuml
import cupy as cp
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.linear_model import Lasso as cuLasso
from cuml.linear_model import Ridge as cuRidge
from cuml.linear_model import ElasticNet as cuElasticNet
from cuml.svm import SVR as cuSVR
from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LassoCV 
from sklearn.feature_selection import SelectFromModel
import os
os.chdir("/local-scratch/GlucoseProject/mobicom23_mobispectral/regression")
import sys
sys.path.append(os.getcwd())
from utils import *
from evaluate import *

# %%
#data_root = '../datasets/dataset_skin/regression'
#file_train = 'concatset_train.txt'
#file_test = 'concatset_test.txt'
#step = 1

# %%
data_root = '../datasets/dataset_glucose/Hyperspectral'
file_train = 'train_data.txt'
file_test = 'test_data.txt'
step = 11

# %%
data_root = '../datasets/dataset_glucose/Hyperspectral'
file_name = 'labels_hs.csv'
step = 1

# %%
X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, step)
# X_train, X_val, y_train, y_val = TTS(X_train, y_train, test_size=0.3, random_state=42)

# %%
X, y, test_data = prepare_data2(data_root, file_name, step)
X_train, X_test, y_train, y_test = TTS(X, y, test_size=0.3, random_state=42)

# %%
# scaler = MinMaxScaler()
scaler = RobustScaler()
# scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# %%
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# %%
# Fit LassoCV model 
lasso_cv = LassoCV(cv=10, random_state=42, verbose=True, alphas=[0.0001, 0.001, 0.01, 0.1, 1, 5], n_jobs=-1)
lasso_cv.fit(X_train, y_train)

print("LassoCV model fit complete...")
# Feature selection 
sfm = SelectFromModel(lasso_cv, prefit=True, threshold=25) 
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
print("Selected Features:", selected_features) 
print("Feature Coefficients:", coefficients) 

# %%
# Step 1, 80 features selected, F score
selected_features_indices = [  1,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,
  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81, 143, 145, 149, 150, 151, 152,
 153, 154, 155, 156, 157, 158, 162, 163]
X_train_selected = X_train[:, selected_features_indices]
# X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)

for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
# Step 1, 50 features selected, F score, standard scaler, with x_val
# Step 1, 50 features selected, F score, robust scaler, without x_val (Same as above!)
selected_features_indices = [ 1, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
 67, 68]
X_train_selected = X_train[:, selected_features_indices]
# X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)

for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
# Step 1, 50 features selected, XGBoost
selected_features_indices = [ 83,  63,  61,  20, 101,  65,   4,   3,   5,  19,   1, 161,  62,   0,   2,  16,   9, 195, 136,  64,  21, 199,  90,  15, 184, 186, 197,   8, 153, 115, 203,  95, 110,  69,   6, 192, 18, 162,  17, 183,  24, 202, 193,  92,  37, 169, 172, 157, 154,  23]
X_train_selected = X_train[:, selected_features_indices]
# X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)


for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
# Step 1, 50 features selected, Lasso
selected_features_indices = [23,  24,  27,  31,  49,  50,  52,  57,  58,  63,  68,  75,  76,  78,  90,  94,  99, 101, 105, 106, 110, 111, 121, 127, 128, 129, 131, 133, 136, 138, 144, 145, 146, 150, 151, 159, 162, 165, 166, 168, 169, 170, 173, 177, 178, 179, 188, 192, 193, 195]
X_train_selected = X_train[:, selected_features_indices]
# X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)


for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
# Step 1, 49 features selected, F score
selected_features_indices = [1, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]
X_train_selected = X_train[:, selected_features_indices]
# X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)

for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
# Step 1, 48 features selected, F score
selected_features_indices = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]
X_train_selected = X_train[:, selected_features_indices]
# X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)

for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
# Step 1, 45 features selected, F score
selected_features_indices = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 41, 42, 43, 44, 45, 46, 47,
 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]
X_train_selected = X_train[:, selected_features_indices]
# X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)


for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
# Step 1, 40 features selected, F score
selected_features_indices = [21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
X_train_selected = X_train[:, selected_features_indices]
# X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)


for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
print(X_train_selected.shape)

# %%
# Step 1, 20 features selected Lasso
selected_features_indices = [ 23,  24,  52,  58,  68,  76,  99, 105, 106, 127, 129, 133, 144, 145, 146, 162, 169, 177,
 179, 195]
X_train_selected = X_train[:, selected_features_indices]
# X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)

for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
# Step 1, Threshold 30, 8 features selected
selected_features_indices = [ 68,  99, 106, 127, 129, 146, 177, 179]
X_train_selected = X_train[:, selected_features_indices]
X_val_selected = X_val[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

test_data_selected = copy.deepcopy(test_data)


for image in test_data_selected:
    test_features = image['sig']
    test_features = scaler.transform(test_features)
    image['sig'] = test_features[:, selected_features_indices]

# %%
for image in test_data:
    image['sig'] = scaler.transform(image['sig'])
    

# %% [markdown]
# ## PCR (PCA + Linear Regression)

# %%
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
pcr = make_pipeline(StandardScaler(), PCA(n_components=25), LinearRegression())
pcr.fit(X_train, y_train)
y_pred_pcr = cp.asnumpy(pcr.predict(X_test))
mse_pcr = mean_squared_error(y_test, y_pred_pcr)
print(f"ARD by signatures: {ARD(y_pred_pcr, y_test).mean()}" )
evaluate_image(test_data, model=pcr)
ard = ard_image(test_data, pcr)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
X_train = cp.array(X_train)
# X_val = cp.array(X_val)
X_test = cp.array(X_test)

# %%
X_train_selected = cp.array(X_train_selected)
# X_val_selected = cp.array(X_val_selected)
X_test_selected = cp.array(X_test_selected)

# %%
# Linear Regression
# 49 bands, F score
lin_reg = cuLinearRegression()
lin_reg.fit(X_train_selected, y_train)
y_pred_lin = cp.asnumpy(lin_reg.predict(X_test_selected))
mse_lin = mean_squared_error(y_test, y_pred_lin)
print(f"ARD by signatures: {ARD(y_pred_lin, y_test).mean()}" )
evaluate_image(test_data_selected, model=lin_reg)
ard = ard_image(test_data_selected, lin_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Linear Regression
# 50 bands, XGBoost
lin_reg = cuLinearRegression()
lin_reg.fit(X_train_selected, y_train)
y_pred_lin = cp.asnumpy(lin_reg.predict(X_test_selected))
mse_lin = mean_squared_error(y_test, y_pred_lin)
print(f"ARD by signatures: {ARD(y_pred_lin, y_test).mean()}" )
evaluate_image(test_data_selected, model=lin_reg)
ard = ard_image(test_data_selected, lin_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Lasso Regression
# 50 bands, F score
lasso_reg = cuLasso()
lasso_reg.fit(X_train_selected, y_train)
y_pred_lasso = cp.asnumpy(lasso_reg.predict(X_test_selected))
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"ARD by signatures: {ARD(y_pred_lasso, y_test).mean()}" )
evaluate_image(test_data_selected, model=lasso_reg)
ard = ard_image(test_data_selected, lasso_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Lasso Regression
# 50 bands, XGBoost
lasso_reg = cuLasso()
lasso_reg.fit(X_train_selected, y_train)
y_pred_lasso = cp.asnumpy(lasso_reg.predict(X_test_selected))
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"ARD by signatures: {ARD(y_pred_lasso, y_test).mean()}" )
evaluate_image(test_data_selected, model=lasso_reg)
ard = ard_image(test_data_selected, lasso_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Ridge Regression
ridge_reg = cuRidge()
ridge_reg.fit(X_train_selected, y_train)
y_pred_ridge = cp.asnumpy(ridge_reg.predict(X_test_selected))
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"ARD by signatures: {ARD(y_pred_ridge, y_test).mean()}" )
evaluate_image(test_data_selected, model=ridge_reg)
ard = ard_image(test_data_selected, ridge_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Elastic Net
elastic_net = cuElasticNet()
elastic_net.fit(X_train_selected, y_train)
y_pred_elastic = cp.asnumpy(elastic_net.predict(X_test_selected))
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
print(f"ARD by signatures: {ARD(y_pred_elastic, y_test).mean()}" )
evaluate_image(test_data_selected, model=elastic_net)
ard = ard_image(test_data_selected, elastic_net)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Multi-Layer Perceptron
mlp_reg = MLPRegressor()
mlp_reg.fit(cp.asnumpy(X_train), cp.asnumpy(y_train))
y_pred_mlp = cp.asnumpy(mlp_reg.predict(X_test))
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
print(f"ARD by signatures: {ARD(y_pred_mlp, y_test).mean()}" )
evaluate_image(test_data, model=mlp_reg)
ard = ard_image(test_data, mlp_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# KNN
knn_reg = cuKNeighborsRegressor()
knn_reg.fit(X_train, y_train)
y_pred_knn = cp.asnumpy(knn_reg.predict(X_test))
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"ARD by signatures: {ARD(y_pred_knn, y_test).mean()}" )
evaluate_image(test_data, model=knn_reg)
ard = ard_image(test_data, knn_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# KNN
# 50 bands, F score
knn_reg = cuKNeighborsRegressor()
knn_reg.fit(X_train_selected, y_train)
y_pred_knn = cp.asnumpy(knn_reg.predict(X_test_selected))
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"ARD by signatures: {ARD(y_pred_knn, y_test).mean()}" )
evaluate_image(test_data_selected, model=knn_reg)
ard = ard_image(test_data_selected, knn_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# KNN
# 50 bands, XGBoost
knn_reg = cuKNeighborsRegressor()
knn_reg.fit(X_train_selected, y_train)
y_pred_knn = cp.asnumpy(knn_reg.predict(X_test_selected))
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f"ARD by signatures: {ARD(y_pred_knn, y_test).mean()}" )
evaluate_image(test_data_selected, model=knn_reg)
ard = ard_image(test_data_selected, knn_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Random Forest
rf_reg = cuRF()
rf_reg.fit(X_train, y_train)
y_pred_rf = cp.asnumpy(rf_reg.predict(X_test))
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"ARD by signatures: {ARD(y_pred_rf, y_test).mean()}" )
evaluate_image(test_data, model=rf_reg)
ard = ard_image(test_data, rf_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Random Forest Selected
# 50 bands, F score
rf_reg = cuRF()
rf_reg.fit(X_train_selected, y_train)
y_pred_rf = cp.asnumpy(rf_reg.predict(X_test_selected))
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"ARD by signatures: {ARD(y_pred_rf, y_test).mean()}" )
evaluate_image(test_data_selected, model=rf_reg)
ard = ard_image(test_data_selected, rf_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Random Forest Selected
# 50 bands, XGBoost
rf_reg = cuRF()
rf_reg.fit(X_train_selected, y_train)
y_pred_rf = cp.asnumpy(rf_reg.predict(X_test_selected))
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"ARD by signatures: {ARD(y_pred_rf, y_test).mean()}" )
evaluate_image(test_data_selected, model=rf_reg)
ard = ard_image(test_data_selected, rf_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# XGBoost
xgb_reg = XGBRegressor(device='cuda', tree_method='hist')
xgb_reg.fit(X_train, y_train)
y_pred_xgb = cp.asnumpy(xgb_reg.predict(X_test))
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"ARD by signatures: {ARD(y_pred_xgb, y_test).mean()}" )
evaluate_image(test_data, model=xgb_reg)
ard = ard_image(test_data, xgb_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# XGBoost selected
# xgb_reg = XGBRegressor(
#             n_estimators=250
#            ,subsample=0.5236842105263158
#            ,random_state=42
#            ,tree_method='hist'
#            ,device='cuda'
#            ,n_job=-1
#            )
xgb_reg = XGBRegressor(tree_method='hist', device='cuda', n_job=-1)
xgb_reg.fit(X_train_selected, y_train)
y_pred_xgb = cp.asnumpy(xgb_reg.predict(X_test_selected))
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"ARD by signatures: {ARD(y_pred_xgb, y_test).mean()}" )
evaluate_image(test_data_selected, model=xgb_reg)
ard = ard_image(test_data_selected, xgb_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# 50 bands, XGBoost
# XGBoost selected
# xgb_reg = XGBRegressor(
#             n_estimators=250
#            ,subsample=0.5236842105263158
#            ,random_state=42
#            ,tree_method='hist'
#            ,device='cuda'
#            ,n_job=-1
#             )
xgb_reg = XGBRegressor(tree_method='hist', device='cuda', n_job=-1)
xgb_reg.fit(X_train_selected, y_train)
y_pred_xgb = cp.asnumpy(xgb_reg.predict(X_test_selected))
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"ARD by signatures: {ARD(y_pred_xgb, y_test).mean()}" )
evaluate_image(test_data_selected, model=xgb_reg)
ard = ard_image(test_data_selected, xgb_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Load XGB model
model = load_model('./Models/XGB_regressor2_selected.model')
y_pred_xgb = cp.asnumpy(model.predict(X_test))
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"ARD by signatures: {ARD(y_pred_xgb, y_test).mean()}" )
evaluate_image(test_data, model=model)
ard = ard_image(test_data, model)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Support Vector Regressor
svr_reg = cuSVR()
svr_reg.fit(X_train, y_train)
y_pred_svr = cp.asnumpy(svr_reg.predict(X_test))
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"ARD by signatures: {ARD(y_pred_svr, y_test).mean()}" )
evaluate_image(test_data, model=svr_reg)
ard = ard_image(test_data, svr_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Support Vector Regressor
# 50 bands, XGBoost
svr_reg = cuSVR()
svr_reg.fit(X_train_selected, y_train)
y_pred_svr = cp.asnumpy(svr_reg.predict(X_test_selected))
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"ARD by signatures: {ARD(y_pred_svr, y_test).mean()}" )
evaluate_image(test_data_selected, model=svr_reg)
ard = ard_image(test_data_selected, svr_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# Support Vector Regressor
# 50 bands, F score
svr_reg = cuSVR()
svr_reg.fit(X_train_selected, y_train)
y_pred_svr = cp.asnumpy(svr_reg.predict(X_test_selected))
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"ARD by signatures: {ARD(y_pred_svr, y_test).mean()}" )
evaluate_image(test_data_selected, model=svr_reg)
ard = ard_image(test_data_selected, svr_reg)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
save_model(svr_reg, './Models/SVR_default_50selected_f.model')

# %%
# Print the Mean Squared Error for each model
print("Mean Squared Error for Linear Regression:", mse_lin)
print("Mean Squared Error for Lasso Regression:", mse_lasso)
print("Mean Squared Error for Ridge Regression:", mse_ridge)
print("Mean Squared Error for Elastic Net:", mse_elastic)
print("Mean Squared Error for MLP:", mse_mlp)
print("Mean Squared Error for SVR:", mse_svr)
print("Mean Squared Error for XGBoost:", mse_xgb)
print("Mean Squared Error for PLS:", mse_pls)

# %%
# # Trained PLS based on S1 and S2
# model = load_model('Models/PLS_regressor1.model')
# evaluate_image(test_data, model=model)

# %%
# PLSR default
plsr_test = PLSRegression()
plsr_test.fit(X_train, y_train)
y_pred_plsr_test = cp.asnumpy(plsr_test.predict(X_test))
mse_plsr_test = mean_squared_error(y_test, y_pred_plsr_test)
print(f"ARD by signatures: {ARD(y_pred_plsr_test, y_test).mean()}" )
evaluate_image(test_data, model=plsr_test)
ard = ard_image(test_data, plsr_test)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
# PLSR
# Best parameter: tol: 1e-06, scale: False, n_components: 20, max_iter: 500
plsr_test = PLSRegression(n_components=25, scale=False, tol=1e-06, max_iter=500)
plsr_test.fit(X_train, y_train)
y_pred_plsr_test = cp.asnumpy(plsr_test.predict(X_test))
mse_plsr_test = mean_squared_error(y_test, y_pred_plsr_test)
print(f"ARD by signatures: {ARD(y_pred_plsr_test, y_test).mean()}" )
evaluate_image(test_data, model=plsr_test)
ard = ard_image(test_data, plsr_test)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

# %%
step3 = np.asarray([ 8, 12, 19, 21, 26, 32, 33, 43, 48, 50, 55, 58, 59, 65]) * 3
step1 = np.asarray([ 24,  52,  58,  68,  76, 99, 105, 106, 127, 129, 144, 145, 146, 162, 177, 179])

step1 = selected_features
#step2 = selected_features * 2
print("step3: ", step3) 
print("step1: ", step1) 

# %%
for image in test_data:
    print(image['label'])


