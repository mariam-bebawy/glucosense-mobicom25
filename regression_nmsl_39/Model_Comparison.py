import time
import copy
import numpy as np
import cuml
import cupy as cp
import xgboost
from xgboost import XGBRegressor, plot_tree, plot_importance
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import graphviz
import pandas as pd
import csv
import os
import sys
import logging

from cuml.svm import SVR as cuSVR
from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression

# Adjust the working directory and system path
os.chdir("/local-scratch/GlucoseProject/mobicom23_mobispectral/regression")
sys.path.append(os.getcwd())

from utils import *
from evaluate import *
from config import *

from utils import initialize_logger

# Initialize logging
log_dir = os.path.join(os.getcwd(), 'experiment_timing.log')
logger = initialize_logger(log_dir)

# Define constants
BANDS_WAVELENGTHS = [397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68, 932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58]

# Subject Indices
SHAP_INDEX = [183, 184, 181, 182, 159, 22, 130, 27, 177, 34, 43, 25, 69, 42, 191, 54, 57, 37, 0, 162, 1, 40, 179, 161, 160, 96, 62, 23, 41, 156, 157, 44, 26, 45, 155, 68, 56, 94, 97, 66, 158, 80, 53, 24, 198, 36, 38, 21, 138, 2, 167, 88, 29, 67, 70, 151, 55, 90, 174, 17, 202, 188, 133, 31, 200, 140, 185, 63, 32, 145, 100, 84, 197, 193, 168, 20, 144, 154, 33, 89, 152, 60, 16, 178, 142, 15, 30, 35, 201, 46, 64, 203, 19, 71, 28, 58, 52, 72, 173, 194, 91, 139, 150, 141, 123, 164, 153, 187, 136, 128, 199, 148, 149, 129, 163, 180, 98, 95, 39, 127, 146, 189, 186, 125, 147, 165, 65, 61, 48, 74, 171, 116, 3, 73, 109, 135, 18, 75, 137, 190, 122, 93, 176, 175, 50, 192, 143, 77, 195, 134, 47, 78, 114, 170, 104, 51, 81, 115, 99, 92, 79, 107, 83, 118, 172, 111, 105, 132, 196, 76, 103, 87, 166, 131, 113, 110, 101, 119, 102, 169, 86, 121, 112, 4, 126, 124, 14, 106, 5, 59, 120, 6, 117, 85, 49, 108, 11, 82, 13, 12, 8, 10, 7, 9]

# Filter subjects (Example IDs)
TEST_SUBJECTS = [10, 18, 29, 5, 7, 13, 22, 31]
TRAIN_SUBJECTS = [1, 2, 3, 4, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 28, 30]

# Define data root and file paths
data_root = '../datasets/dataset_glucose/mobile/GooglePixel4XL'
file_train = 'mobi_train.csv'
file_test = 'mobi_test.csv'
step = 1

# Prepare data
logger.info('Preparing data...')
X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, 'mobi', step=step)

# Initialize scalers
x_scaler = RobustScaler()
y_scaler = RobustScaler()

# Scale the data
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

print(f"Data shapes => X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

logger.info('Starting model training and timing experiments...')

# Dictionary to store timing results
timing_results = {
    'Model': [],
    'Training_Time_sec': [],
    'Prediction_Time_sec': []
}

###########################   Models Timing Comparison   ###########################

# 1. XGBoost Default Settings
print("XGBoost Default Settings")
model = XGBRegressor(
    tree_method='hist',
    device='cuda',
    n_jobs=-1
)

# Measure training time
train_start = time.perf_counter()
model.fit(X_train, y_train)
train_end = time.perf_counter()
train_time = train_end - train_start

# Measure prediction time
predict_start = time.perf_counter()
y_pred = cp.asnumpy(model.predict(X_test))
predict_end = time.perf_counter()
predict_time = predict_end - predict_start

# Store results
timing_results['Model'].append('XGBoost_Default')
timing_results['Training_Time_sec'].append(train_time)
timing_results['Prediction_Time_sec'].append(predict_time)

print(f"Training Time: {train_time:.4f} seconds")
print(f"Prediction Time: {predict_time:.4f} seconds\n")

# 2. XGBoost Tuned Settings
print("XGBoost Tuned Settings")
model = XGBRegressor(
    n_estimators=800,
    subsample=0.894736842105263,
    random_state=42,
    tree_method='hist',
    device='cuda',
    n_jobs=-1,
    learning_rate=0.26,
    gamma=0.006,
    max_depth=8
)

# Measure training time
train_start = time.perf_counter()
model.fit(X_train, y_train)
train_end = time.perf_counter()
train_time = train_end - train_start

# Measure prediction time
predict_start = time.perf_counter()
y_pred = cp.asnumpy(model.predict(X_test))
predict_end = time.perf_counter()
predict_time = predict_end - predict_start

# Store results
timing_results['Model'].append('XGBoost_Tuned')
timing_results['Training_Time_sec'].append(train_time)
timing_results['Prediction_Time_sec'].append(predict_time)

print(f"Training Time: {train_time:.4f} seconds")
print(f"Prediction Time: {predict_time:.4f} seconds\n")

# 3. Random Forest
print("Random Forest")
model = cuRF(random_state=42)

# Measure training time
train_start = time.perf_counter()
model.fit(X_train, y_train)
train_end = time.perf_counter()
train_time = train_end - train_start

# Measure prediction time
predict_start = time.perf_counter()
y_pred = cp.asnumpy(model.predict(X_test))
predict_end = time.perf_counter()
predict_time = predict_end - predict_start

# Store results
timing_results['Model'].append('Random_Forest')
timing_results['Training_Time_sec'].append(train_time)
timing_results['Prediction_Time_sec'].append(predict_time)

print(f"Training Time: {train_time:.4f} seconds")
print(f"Prediction Time: {predict_time:.4f} seconds\n")

# 4. K-Nearest Neighbors
print("K-Nearest Neighbors")
model = cuKNeighborsRegressor()

# Measure training time
train_start = time.perf_counter()
model.fit(X_train, y_train)
train_end = time.perf_counter()
train_time = train_end - train_start

# Measure prediction time
predict_start = time.perf_counter()
y_pred = cp.asnumpy(model.predict(X_test))
predict_end = time.perf_counter()
predict_time = predict_end - predict_start

# Store results
timing_results['Model'].append('KNN')
timing_results['Training_Time_sec'].append(train_time)
timing_results['Prediction_Time_sec'].append(predict_time)

print(f"Training Time: {train_time:.4f} seconds")
print(f"Prediction Time: {predict_time:.4f} seconds\n")

# 5. Partial Least Squares
print("Partial Least Squares (PLS)")
model = PLSRegression()

# Measure training time
train_start = time.perf_counter()
model.fit(X_train, y_train)
train_end = time.perf_counter()
train_time = train_end - train_start

# Measure prediction time
predict_start = time.perf_counter()
y_pred = model.predict(X_test)
predict_end = time.perf_counter()
predict_time = predict_end - predict_start

# Store results
timing_results['Model'].append('PLSRegression')
timing_results['Training_Time_sec'].append(train_time)
timing_results['Prediction_Time_sec'].append(predict_time)

print(f"Training Time: {train_time:.4f} seconds")
print(f"Prediction Time: {predict_time:.4f} seconds\n")

# 6. Linear Support Vector Regressor
print("Linear Support Vector Regressor (LinearSVR)")
model = LinearSVR(random_state=42)

# Measure training time
train_start = time.perf_counter()
model.fit(X_train, y_train)
train_end = time.perf_counter()
train_time = train_end - train_start

# Measure prediction time
predict_start = time.perf_counter()
y_pred = model.predict(X_test)
predict_end = time.perf_counter()
predict_time = predict_end - predict_start

# Store results
timing_results['Model'].append('LinearSVR')
timing_results['Training_Time_sec'].append(train_time)
timing_results['Prediction_Time_sec'].append(predict_time)

print(f"Training Time: {train_time:.4f} seconds")
print(f"Prediction Time: {predict_time:.4f} seconds\n")

# Compile and display timing results
timing_df = pd.DataFrame(timing_results)
print("=== Training and Prediction Times ===")
print(timing_df)

# Optionally, save the timing results to a CSV file
output_path = './Experiment/timing_results.csv'
timing_df.to_csv(output_path, index=False)
print(f"\nTiming results saved to {output_path}")

logger.info('Model timing experiments completed and results saved.')
