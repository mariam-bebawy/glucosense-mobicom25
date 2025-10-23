
import os
import copy
import argparse
import numpy as np
from model import Regressors

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV 
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

import os
os.chdir("/local-scratch/GlucoseProject/mobicom23_mobispectral/regression")
import sys
sys.path.append(os.getcwd())
from utils import prepare_data, evaluate_image, load_model, save_model

from utils import *
# logging
log_dir = os.path.join(os.getcwd(), 'train.log')
logger = initialize_logger(log_dir)

X_train, y_train, X_test, y_test, test_data = prepare_data('../datasets/dataset_skin/regression', 'concatset_train.txt', 'concatset_test.txt', 1)
# X_train, y_train, X_test, y_test, test_data = prepare_data('../datasets/dataset_skin/regression', 'labels_s2_train.txt', 'labels_s2_test.txt', 3)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_val, y_val = None, None

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Show dimensions of data before LassoCV
print("X_train shape: ", X_train.shape)
# print("X_val shape: ", X_val.shape)
print("X_test shape: ", X_test.shape)

print("Data preparation complete...")

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

print("### Training models and performing hyperparameter search...")

# # MLP
# regressor_obj = Regressors(os.path.join('../Models', 'MLP_regressor2'))
# regressor_obj.model_search('MLP', X_train, y_train, X_val, y_val, X_test, y_test)

# # Random Forest
# regressor_obj = Regressors(os.path.join('Models', 'RF_regressor4_selected'))
# model = regressor_obj.model_search('RF', X_train_selected, y_train, X_test_selected, y_test, X_test_selected, y_test, gpu=True)

# Linear Regression
# regressor_obj = Regressors(os.path.join('Models', 'LR_regressor3'))
# model = regressor_obj.model_search('LR', X_train, y_train, X_val, y_val, X_test, y_test, gpu=True)

# # SVR
# regressor_obj = Regressors(os.path.join('Models', 'SVR_regressor2_selected'))
# model = regressor_obj.model_search('SVR', X_train_selected, y_train, X_test_selected, y_test, X_test_selected, y_test, gpu=True)

# # PLS  Hyperparameter searched on S1 and S2
# regressor_obj = Regressors(os.path.join('Models', 'PLS_regressor1'))
# regressor_obj.model_search('PLS', X_train, y_train, X_val, y_val, X_test, y_test)

# XGB 
regressor_obj = Regressors(os.path.join('Models', 'XGB_regressor3_selected_50band'))
model = regressor_obj.model_search('XGB', X_train_selected, y_train, X_test_selected, y_test, X_test_selected, y_test, gpu=True)

# # Ridge Regression
# regressor_obj = Regressors(os.path.join('Models', 'RR_regressor1'))
# model = regressor_obj.model_search('RR', X_train, y_train, X_val, y_val, X_test, y_test, gpu=True)

# # Lasso Regression
# regressor_obj = Regressors(os.path.join('Models', 'LAS_regressor1'))
# model = regressor_obj.model_search('LAS', X_train, y_train, X_val, y_val, X_test, y_test, gpu=True)


# # Elastic Net
# regressor_obj = Regressors(os.path.join('Models', 'EN_regressor2'))
# model = regressor_obj.model_search('EN', X_train, y_train, X_val, y_val, X_test, y_test, gpu=True)


# model = load_model('Models/SVR_regressor2_selected.model')
# evaluate_image(test_data_selected, model=model)
# ard = ard_image(test_data_selected, model)
# print("ARD by images: ", ard)
# print("Average ARD: ", ard.mean())