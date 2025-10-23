import copy
import numpy as np
import cuml
import cupy as cp
import xgboost
from xgboost import XGBRegressor
from xgboost import plot_tree, plot_importance
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as TTS
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import graphviz
import time

from cuml.svm import SVR as cuSVR
from cuml.ensemble import RandomForestRegressor as cuRF
from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression

from MLP import MLPRegressorPyTorch
import VGG

import os
os.chdir("/local-scratch/GlucoseProject/mobicom23_mobispectral/regression")
import sys
sys.path.append(os.getcwd())
from utils import *
from evaluate import *
from config import *

from utils import initialize_logger
# logging
log_dir = os.path.join(os.getcwd(), 'experiment.log')
logger = initialize_logger(log_dir)

BANDS_WAVELENGTHS = [397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68, 932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58]
# index = [183 184 181 182 159 22 130 27 177 34 43 25 69 42 191 54 57 37  0 162 1 40 179 161 160 96 62 23 41 156 157 44 26 45 155 68  56 94 97 66 158 80 53 24 198 36 38 21 138 2 167 88 29 67  70 151 55 90 174 17 202 188 133 31 200 140 185 63 32 145 100 84  197 193 168 20 144 154 33 89 152 60 16 178 142 15 30 35 201 46  64 203 19 71 28 58 52 72 173 194 91 139 150 141 123 164 153 187  136 128 199 148 149 129 163 180 98 95 39 127 146 189 186 125 147 165  65 61 48 74 171 116 3 73 109 135 18 75 137 190 122 93 176 175  50 192 143 77 195 134 47 78 114 170 104 51 81 115 99 92 79 107  83 118 172 111 105 132 196 76 103 87 166 131 113 110 101 119 102 169  86 121 112 4 126 124 14 106 5 59 120 6 117 85 49 108 11 82  13 12 8 10 7 9]
SHAP_INDEX = [183, 184, 181, 182, 159, 22, 130, 27, 177, 34, 43, 25, 69, 42, 191, 54, 57, 37, 0, 162, 1, 40, 179, 161, 160, 96, 62, 23, 41, 156, 157, 44, 26, 45, 155, 68, 56, 94, 97, 66, 158, 80, 53, 24, 198, 36, 38, 21, 138, 2, 167, 88, 29, 67, 70, 151, 55, 90, 174, 17, 202, 188, 133, 31, 200, 140, 185, 63, 32, 145, 100, 84, 197, 193, 168, 20, 144, 154, 33, 89, 152, 60, 16, 178, 142, 15, 30, 35, 201, 46, 64, 203, 19, 71, 28, 58, 52, 72, 173, 194, 91, 139, 150, 141, 123, 164, 153, 187, 136, 128, 199, 148, 149, 129, 163, 180, 98, 95, 39, 127, 146, 189, 186, 125, 147, 165, 65, 61, 48, 74, 171, 116, 3, 73, 109, 135, 18, 75, 137, 190, 122, 93, 176, 175, 50, 192, 143, 77, 195, 134, 47, 78, 114, 170, 104, 51, 81, 115, 99, 92, 79, 107, 83, 118, 172, 111, 105, 132, 196, 76, 103, 87, 166, 131, 113, 110, 101, 119, 102, 169, 86, 121, 112, 4, 126, 124, 14, 106, 5, 59, 120, 6, 117, 85, 49, 108, 11, 82, 13, 12, 8, 10, 7, 9]

# # print the bands
# for i in SHAP_INDEX:
#     print(BANDS_WAVELENGTHS[i])

# with open('./Experiment/xgb.fmap', 'w') as file:
#     for idx, wavelength in enumerate(BANDS_WAVELENGTHS):
#         file.write(f'{idx}\t{wavelength}\tq\n')

# # Filter the subject id to be used
# # Test subject IDs 10, 18, 29, 5, 7, 13, 22, 31. Remaining in train
# TEST_SUBJECTS = [10, 18, 29, 5, 7, 13, 22, 31]
# TRAIN_SUBJECTS = [1, 2, 3, 4, 6, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 28, 30]
# filter_rows_by_id('../datasets/dataset_glucose/mobile/GooglePixel4XL/labels_GooglePixel.csv', '../datasets/dataset_glucose/mobile/GooglePixel4XL/test_filtered.csv', TEST_SUBJECTS)
# # Filter the train data
# filter_rows_by_id('../datasets/dataset_glucose/mobile/GooglePixel4XL/labels_GooglePixel.csv', '../datasets/dataset_glucose/mobile/GooglePixel4XL/train_filtered.csv', TRAIN_SUBJECTS)

# data_root = '../datasets/dataset_glucose/Hyperspectral'
# file_train = 'train_data.csv'
# file_test = 'test_data.csv'
# step = 1

# # Google Pixel 4XL
# data_root = '../datasets/dataset_glucose/mobile/GooglePixel4XL'
# labels_file = 'labels_GooglePixel.csv'
# file_train = 'train_filtered.csv'
# file_test = 'test_filtered.csv'
# step = 1

data_root = '../datasets/dataset_glucose/mobile/GooglePixel4XL'
file_train = 'mobi_train.csv'
file_test = 'mobi_test.csv'
step = 1

# data_root = '../datasets/dataset_glucose/mobile/RaspberryPi-RGB+ToF'
# file_train = 'mobi_train.csv'
# file_test = 'mobi_test.csv'
# step = 1

# # %%
# data_root = '../datasets/dataset_glucose/mobile/OnSemi-custom-RGB-NIR'
# file_train = 'mobi_train.csv'
# file_test = 'mobi_test.csv'
# step = 1

logger.info('Preparing data...')

# def load_data_from_csv(file_path):
#     # Load the CSV file into a DataFrame
#     df = pd.read_csv(file_path)
    
#     # Extract the prediction and reference values
#     pred_values = df['Prediction'].values
#     ref_values = df['Glucose(mg/dL) - Ref'].values
    
#     return pred_values, ref_values


# # file_path = '/local-scratch/GlucoseProject/mobicom23_mobispectral/datasets/dataset_glucose/mobile/GooglePixel4XL/test.csv'
# file_path = '/local-scratch/GlucoseProject/mobicom23_mobispectral/datasets/dataset_glucose/mobile/RaspberryPi-RGB+ToF/test.csv'
# pred, ref = load_data_from_csv(file_path)

# X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, 'hyper', index=index)
# X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, 'hyper', step=step)

# X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, 'mobi', step=step)
X_train, y_train, X_test, y_test, test_data = prepare_data_fixed_size(data_root, file_train, file_test, 'mobi', step=step)

# # # plt.figure()

# # # for row in X_train:
# # #     plt.plot(row)

# # # plt.savefig('spectral_signal.png')

# # # print('Saved graph of spectral signal')

# # # scaler = MinMaxScaler()
# x_scaler = RobustScaler()
# y_scaler = RobustScaler()
# # # x_scaler = StandardScaler()
# # # y_scaler = StandardScaler()
# # # scaler = StandardScaler()
# X_train = x_scaler.fit_transform(X_train)
# # X_val = scaler.transform(X_val)
# X_test = x_scaler.transform(X_test)
# y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
# y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


logger.info('Model fitting...')

# # XGBoost
# xgb_reg = XGBRegressor(n_estimators=800
#                     ,subsample=0.894736842105263
#                     ,random_state=42
#                     ,tree_method='hist'
#                     ,device='cuda'
#                     ,n_job=-1
#                     ,learning_rate=0.26
#                     ,gamma=0.006
#                     ,max_depth=8)

# xgb_reg = XGBRegressor(n_estimators=800
#                     ,subsample=0.894736842105263
#                     ,random_state=42
#                     ,tree_method='hist'
#                     ,device='cuda'
#                     ,n_job=-1
#                     ,learning_rate=0.26
#                     ,gamma=0.006
#                     ,max_depth=3)
# # # xgb_reg = XGBRegressor(random_state=42, n_jobs=-1, device='cuda', tree_method='hist')

# xgb_reg.fit(X_train, y_train)

# model = xgb_reg
# # save_model(model, './Models/XGB_HS_Jan26.pkl')
# # # model = load_model('./Models/XGB_HS_depth2.pkl')
# y_pred = cp.asnumpy(model.predict(X_test))
# print(f"ARD by signatures: {ARD(y_pred, y_test).mean()}" )
# ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
# print("ARD by images: ", ard)
# average = ard.mean()
# print(f"Average ARD: {average}")
# logging.info(f"Average ARD: {average}")
# evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, save_path='./Experiment/CEG_xgb_GooglePixel_filtered.png')
# evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, save_path='./Experiment/SEG_xgb_GooglePixel_filtered.png')
# r, p = evaluate_image_PCC(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, save_path='./Experiment/PCC_xgb_GooglePixel_filtered.png')
# print(f"r: {r}, p-value: {p}")
# logger.info(f"r: {r}, p-value: {p}")
# CDF(ard, save_path='./Experiment/CDF_xgb_GooglePixel_filtered.png')
# evaluate_image_Overlapped_PEG_SEG(test_data, model=model, diabetes_type=1, scaler_X=x_scaler, scaler_y=y_scaler, show=False, save_path='./Experiment/Overlapped_PEG_SEG_xgb_GooglePixel_filtered.png')

# MLP
# mlp_reg = MLPRegressorPyTorch(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', alpha=0.001,
#                  batch_size=409600, learning_rate='constant', learning_rate_init=0.001, max_iter=200,
#                  shuffle=True, random_state=42, tol=1e-4, verbose=True, warm_start=False,
#                  momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1,
#                  beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=5, max_fun=15000, dropout_rate=0.2)

# # Sklearn MLP
# mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', alpha=0.001,
#                  batch_size=409600, learning_rate='constant', learning_rate_init=0.001, max_iter=200,
#                  shuffle=True, random_state=42, tol=1e-4, verbose=True, warm_start=False,
#                  momentum=0.9, nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1,
#                  beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=5)

# VGG
channels = X_train.shape[1] # X_train: (1383, 50, 64, 64)
input_size = X_train.shape[2:]

model = VGG.VGG19(input_channels=channels, input_size=input_size)

# Change the 

# Record the training and prediction time
train_start = time.perf_counter()
VGG.train(model, X_train, y_train, X_test, y_test, epochs=200, lr=1e-6, patience=10)
train_end = time.perf_counter()
train_time = train_end - train_start

# Save the model
save_model(model, './Models/VGG_GooglePixel_Feb24.pkl')

predict_start = time.perf_counter()
y_pred = model.predict(X_test)
predict_end = time.perf_counter()

predict_time = predict_end - predict_start
print(f"Training time: {train_time} seconds")
print(f"Prediction time: {predict_time} seconds")
ard = ard_image(test_data, model)
print("Average ARD: ", ard.mean())
plt2, zone = evaluate_image_CEG(test_data, model=model, show=False)
print("Clark Error Grid Zone: ", zone)
risk_levels_percentage = evaluate_image_SEG(test_data, model=model, show=False)
print("Surveillance Error Grid Zone: ", risk_levels_percentage)

# # Plot the feature importance
# # Visualize feature importance
# plt.figure(figsize=(20, 10))
# plt.bar(range(0,len(BANDS_WAVELENGTHS),1), xgb_reg.feature_importances_)
# plt.xlabel("Feature index")
# plt.ylabel("Feature importance")
# plt.xticks(range(0, 204, 5), BANDS_WAVELENGTHS[0::5], rotation=90)
# plt.title("Feature importance of all features")
# plt.savefig('./Experiment/Feature_importance.png')

# # xgb_reg.get_booster().feature_names = [str(band) for band in BANDS_WAVELENGTHS]

# # Plot the importance of the features using built-in function
# ax = plot_importance(xgb_reg, max_num_features=20, title='Band Importance', ylabel='Wavelength (nm)', xlabel='Frequency', fmap='./Experiment/xgb.fmap')  
# plt.savefig('./Experiment/Feature_importance_built-in.png')

# # Plot tree structure using xgboost its own function
# plt.figure(figsize=(20, 10))
# plot_tree(xgb_reg, num_trees=0, fmap='./Experiment/xgb.fmap', rankdir='TB')
# # plt.savefig('./Experiment/XGB_HS.png')
# # Save the figure into svg file
# plt.savefig('../graph/XGBoost/tree_structure_HS_full_Jan26.svg', format='svg')
# # Plot tree structure using graphviz
# tree_dot = xgboost.to_graphviz(xgb_reg, num_trees=2)

# # Save the dot file
# dot_file_path = "../graph/XGBoost/tree_structure_HS_full_Jan26.dot"
# tree_dot.save(dot_file_path)

# # Create a Graphviz Source object
# graph = graphviz.Source.from_file(dot_file_path)

# # Define the output path without extension
# output_path = "../graph/XGBoost/tree_structure_HS_full_Jan26_graphviz"

# # Render the graph to SVG
# # graph.render(output_path, format='png', cleanup=True)
# graph.render(output_path, format='svg', cleanup=True)

# logger.info('Cross validation...')

# # Google Pixel 4XL
# data_root = '../datasets/dataset_glucose/mobile/GooglePixel4XL'
# labels_file = 'labels_GooglePixel.csv'
# file_train = 'train_temp.csv'
# file_test = 'test_temp.csv'
# step = 1

# # Without calibration
# SUB, ARD, ZONE_CLARK, ZONE_SURVEILLANCE = cross_validate(model=xgb_reg, data_root=data_root, labels_file=labels_file, temp_train_file=file_train, temp_test_file=file_test, step=step, dataset='mobi', show=False, save_path='../graph/XGBoost/ARD_CrossSubjects_GooglePixel.png')
# with open('./Experiment/LeaveOneOut/LeaveOneOut_GooglePixel.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Subject', 'ARD', 'Clark Error Grid', 'Surveillance Error Grid'])
#     for i in range(len(SUB)):
#         writer.writerow([SUB[i], ARD[i], ZONE_CLARK[i], ZONE_SURVEILLANCE[i]])
        

# # With calibration
# SUB, ARD, ZONE_CLARK, ZONE_SURVEILLANCE = cross_validate_calibration(model=xgb_reg, data_root=data_root, labels_file=labels_file, temp_train_file=file_train, temp_test_file=file_test, step=step, dataset='mobi', show=False, save_path='../graph/XGBoost/ARD_CrossSubjects_GooglePixel_Calibration_2.pdf')
# with open('./Experiment/LeaveOneOut/LeaveOneOut_Calibration_3_GooglePixel.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Subject', 'ARD', 'Clark Error Grid', 'Surveillance Error Grid'])
#     for i in range(len(SUB)):
#         writer.writerow([SUB[i], ARD[i], ZONE_CLARK[i], ZONE_SURVEILLANCE[i]])

# # Raspberry Pi RGB+ToF
# data_root = '../datasets/dataset_glucose/mobile/RaspberryPi-RGB+ToF'
# labels_file = 'labels_RaspberryPi-RGB+ToF.csv'
# file_train = 'train_temp.csv'
# file_test = 'test_temp.csv'
# step = 1

# # Without calibration
# SUB, ARD, ZONE_CLARK, ZONE_SURVEILLANCE = cross_validate(model=xgb_reg, data_root=data_root, labels_file=labels_file, temp_train_file=file_train, temp_test_file=file_test, step=step, dataset='mobi', show=False, save_path='../graph/XGBoost/ARD_CrossSubjects_ToF.png')
# # Record the information of each subject into a csv file
# with open('./Experiment/LeaveOneOut/LeaveOneOut_ToF_Jan5.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Subject', 'ARD', 'Clark Error Grid', 'Surveillance Error Grid'])
#     for i in range(len(SUB)):
#         writer.writerow([SUB[i], ARD[i], ZONE_CLARK[i], ZONE_SURVEILLANCE[i]])

# # With calibration
# SUB, ARD, ZONE_CLARK, ZONE_SURVEILLANCE = cross_validate_calibration(model=xgb_reg, data_root=data_root, labels_file=labels_file, temp_train_file=file_train, temp_test_file=file_test, step=step, dataset='mobi', show=False, save_path='../graph/XGBoost/ARD_CrossSubjects_ToF_Calibration.pdf')
# with open('./Experiment/LeaveOneOut/LeaveOneOut_Calibration_ToF.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Subject', 'ARD', 'Clark Error Grid', 'Surveillance Error Grid'])
#     for i in range(len(SUB)):
#         writer.writerow([SUB[i], ARD[i], ZONE_CLARK[i], ZONE_SURVEILLANCE[i]])

# # Raspberry Pi RGB-NoIR
# data_root = '../datasets/dataset_glucose/mobile/RaspberryPi-RGB-NoIR'
# labels_file = 'labels_RaspberryPi-RGB_NoIR.csv'
# file_train = 'train_temp.csv'
# file_test = 'test_temp.csv'
# step = 1

# SUB, ARD, ZONE_CLARK, ZONE_SURVEILLANCE = cross_validate(model=xgb_reg, data_root=data_root, labels_file=labels_file, temp_train_file=file_train, temp_test_file=file_test, step=step, dataset='mobi', show=False, save_path='../graph/XGBoost/ARD_CrossSubjects_RasPi_NoIR.png')
# with open('./Experiment/LeaveOneOut/LeaveOneOut_RasPi_NoIR.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Subject', 'ARD', 'Clark Error Grid', 'Surveillance Error Grid'])
#     for i in range(len(SUB)):
#         writer.writerow([SUB[i], ARD[i], ZONE_CLARK[i], ZONE_SURVEILLANCE[i]])

# # OnSemi
# data_root = '../datasets/dataset_glucose/mobile/OnSemi-custom-RGB-NIR'
# labels_file = 'labels_OnSemi.csv'
# file_train = 'train_temp.csv'
# file_test = 'test_temp.csv'
# step = 1

# SUB, ARD, ZONE_CLARK, ZONE_SURVEILLANCE = cross_validate(model=xgb_reg, data_root=data_root, labels_file=labels_file, temp_train_file=file_train, temp_test_file=file_test, step=step, dataset='mobi', show=False, save_path='../graph/XGBoost/ARD_CrossSubjects_Custom.png')
# with open('./Experiment/LeaveOneOut/LeaveOneOut_Custom.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Subject', 'ARD', 'Clark Error Grid', 'Surveillance Error Grid'])
#     for i in range(len(SUB)):
#         writer.writerow([SUB[i], ARD[i], ZONE_CLARK[i], ZONE_SURVEILLANCE[i]])


# zones = parkeszones(2, pred, ref, units="mgdl", numeric=True)
# parkes(2, pred, ref, units="mgdl", color_points="auto", percentage=True)
# plt.savefig('parkes.png')

# plot_overlapped_PEG_SEG(pred, ref, 2, seg_img='/local-scratch/GlucoseProject/mobicom23_mobispectral/graph/SEG_background/seg600.png', show=False, save_path='Overlapped_PEG_SEG_2.png')
# plot_overlapped_PEG_SEG2(pred, ref, img='/local-scratch/GlucoseProject/mobicom23_mobispectral/graph/SEG_background/seg600.png', show=False, save_path='Overlapped_PEG_SEG_2.png')

# plot_parkes_error_grid2(pred, ref, 1, show=False, save_path='parkes.png')

# plot_parkes_error_grid(pred, ref, 1, show=False, save_path='parkes.png')
# plt, zone = clarke_error_grid(pred, ref, 'Clark Error Grid')
# plt.savefig('clarke.png')

# # MAE vs Reference
# MAE = np.abs(pred - ref)


# # plt.figure(figsize=(10, 6))
# # plt.scatter(ref, MAE, marker='o', color='b')

# # # Add labels and title
# # plt.xlabel("Measured Blood Glucose Values (mg/dL)")
# # plt.ylabel("Absolute Error (mg/dL)")
# # plt.title("Absolute Error rises with higher Measured Blood Glucose Values")
# # plt.savefig('MAE_plot.png')

# # Create a DataFrame for easier manipulation
# df = pd.DataFrame({
#     'Measured Blood Glucose Values (mg/dL)': ref,
#     'Mean Absolute Error': MAE
# })

# # Group by 'Measured Blood Glucose Values (mg/dL)' and calculate the mean MAE
# grouped_df = df.groupby('Measured Blood Glucose Values (mg/dL)').mean().reset_index()

# # Sort the DataFrame by 'Measured Blood Glucose Values (mg/dL)'
# grouped_df = grouped_df.sort_values(by='Measured Blood Glucose Values (mg/dL)')

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(grouped_df['Measured Blood Glucose Values (mg/dL)'],
#          grouped_df['Mean Absolute Error'],
#          marker='o', linestyle='-', color='b')

# # Add labels and title
# plt.xlabel("Measured Blood Glucose Values (mg/dL)")
# plt.ylabel("Mean Absolute Error (mg/dL)")
# plt.title("Mean Absolute Error rises with higher glucose values")

# plt.savefig('MAE_plot.png')

# def load_data_from_csv(file_path):
#     # Load the CSV file into a DataFrame
#     df = pd.read_csv(file_path)
    
#     # Extract the prediction and reference values for each camera
#     cameras = ['$Pixel_{RGB+NIR}$', '$RasPi_{RGB+ToF}$', '$Custom_{RGB-NIR}$', '$RasPi_{RGB-NoIR}$']
#     data = {}
    
#     for camera in cameras:
#         ref_col = f'Ref {camera}'
#         pred_col = f'Pred {camera}'
        
#         if ref_col in df.columns and pred_col in df.columns:
#             ref_values = df[ref_col].values
#             pred_values = df[pred_col].values

# #             MAE = np.abs(ref_values - pred_values)

# #             data[camera] = {
# #                 'Measured Blood Glucose Values (mg/dL)': ref_values,
# #                 'Mean Absolute Error': MAE
# #             }
#             data[camera] = {
#                 'Measured Blood Glucose Values (mg/dL)': ref_values,
#                 'Prediction': pred_values
#             }
    
#     return data

# # Load data
# file_path = '/local-scratch/GlucoseProject/mobicom23_mobispectral/datasets/dataset_glucose/mobile/RaspberryPi-RGB+ToF/test.csv'
# data = load_data_from_csv(file_path)

# # Plot
# plt.figure(figsize=(12, 8))

# for camera, values in data.items():
#     ref_values = values['Measured Blood Glucose Values (mg/dL)']
#     # mae_values = values['Mean Absolute Error']
#     pred_values = values['Prediction']

#     # Calculate the percentage of values fall in each zone in Parkes Error Grid
#     zone = calculate_zone_percentages_PEG(ref_values, pred_values, diabetes_type=1)
#     logging.info(f"Parkes Error Grid for {camera}:")
#     logging.info(f"Zone A: {zone['A']:.2f}%")
#     logging.info(f"Zone B: {zone['B']:.2f}%")
#     logging.info(f"Zone C: {zone['C']:.2f}%")
#     logging.info(f"Zone D: {zone['D']:.2f}%")
#     logging.info(f"Zone E: {zone['E']:.2f}%")
#     logging.info('')
    
#     # Group by the measured glucose values and calculate mean MAE
#     df_camera = pd.DataFrame({
#         'Measured Blood Glucose Values (mg/dL)': ref_values,
#         'Mean Absolute Error': mae_values
#     })
    
#     grouped_df = df_camera.groupby('Measured Blood Glucose Values (mg/dL)').mean().reset_index()
#     grouped_df = grouped_df.sort_values(by='Measured Blood Glucose Values (mg/dL)')
    
#     # Plot
#     plt.plot(grouped_df['Measured Blood Glucose Values (mg/dL)'],
#              grouped_df['Mean Absolute Error'],
#              marker='o',
#              linestyle='-',
#              label=camera)

# # Add labels, title, and legend
# plt.xlabel("Measured Blood Glucose Values (mg/dL)")
# plt.ylabel("Mean Absolute Error")
# plt.title("Mean Absolute Error vs. Measured Blood Glucose Values for Different Cameras")
# plt.legend(title='Camera Type')
# plt.savefig('MAE_plot.png')

# Number of bands vs MARD
# mard = []
# RANGE = range(1, 205)
# Best_MARD = np.inf
# Best_Bands = -1
# for i in RANGE:
#     index_subset = SHAP_INDEX[:i]
#     X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, 'hyper', index=index_subset)
    
#     x_scaler = RobustScaler()
#     y_scaler = RobustScaler()
#     X_train = x_scaler.fit_transform(X_train)
#     X_test = x_scaler.transform(X_test)
#     y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
#     y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

#     logging.info(f"Number of bands: {i}")
#     logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

#     xgb_reg = XGBRegressor(n_estimators=800
#                         ,subsample=0.894736842105263
#                         ,random_state=42
#                         ,tree_method='hist'
#                         ,device='cuda'
#                         ,n_job=-1
#                         ,learning_rate=0.26
#                         ,gamma=0.006
#                         ,max_depth=8)

#     xgb_reg.fit(X_train, y_train)
#     y_pred_xgb = cp.asnumpy(xgb_reg.predict(X_test))
#     ard = ard_image(test_data, xgb_reg, scaler_X=x_scaler, scaler_y=y_scaler)
#     average = ard.mean()
#     logging.info(f"Average ARD: {average}")
#     mard.append(average)
#     if Best_MARD > average:
#         Best_MARD = average
#         Best_Bands = i

# logging.info(f"Best MARD: {Best_MARD}, Best number of bands: {Best_Bands}")
# np.save('./Experiment/MARD_vs_Bands.npy', mard)
# mard = np.load('./Experiment/MARD_vs_Bands.npy')
# plt.figure()
# plt.plot(RANGE, mard)
# plt.xlabel("Number of bands")
# plt.ylabel("MARD")
# # X ticks should show every 10 bands
# plt.xticks(range(9, 204, 10), range(10, 205, 10), rotation=90)
# plt.savefig('./Experiment/MARD_vs_Bands.png')

###########################   Models Comparison   ###########################
# # # XGBoost Default Settings
# print("XGBoost Default Settings")
# model = XGBRegressor(tree_method='hist'
#                         ,device='cuda'
#                         ,n_job=-1)
# model.fit(X_train, y_train)
# y_pred_xgb = cp.asnumpy(model.predict(X_test))
# # print(f"ARD by signatures: {ARD(y_pred_xgb, y_test).mean()}" )
# ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
# print("ARD by images: ", ard)
# print("Average ARD: ", ard.mean())
# plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Clark Error Grid Zone: ", zone)
# risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Surveillance Error Grid Zone: ", risk_levels_percentage)


# # # XGBoost Tuned Settings
# print("XGBoost Tuned Settings")
# model = XGBRegressor(n_estimators=800
#                         ,subsample=0.894736842105263
#                         ,random_state=42
#                         ,tree_method='hist'
#                         ,device='cuda'
#                         ,n_job=-1
#                         ,learning_rate=0.26
#                         ,gamma=0.006
#                         ,max_depth=8)
# model.fit(X_train, y_train)
# y_pred_xgb = cp.asnumpy(model.predict(X_test))
# # print(f"ARD by signatures: {ARD(y_pred_xgb, y_test).mean()}" )
# ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
# print("ARD by images: ", ard)
# print("Average ARD: ", ard.mean())
# plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Clark Error Grid Zone: ", zone)
# risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Surveillance Error Grid Zone: ", risk_levels_percentage)

# # Random Forest
# print("Random Forest")
# model = cuRF(random_state=42)
# model.fit(X_train, y_train)
# y_pred_rf = cp.asnumpy(model.predict(X_test))
# ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
# print("ARD by images: ", ard)
# print("Average ARD: ", ard.mean())
# plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Clark Error Grid Zone: ", zone)
# risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Surveillance Error Grid Zone: ", risk_levels_percentage)

# # KNN
# print("KNN")
# model = cuKNeighborsRegressor()
# model.fit(X_train, y_train)
# y_pred_knn = cp.asnumpy(model.predict(X_test))
# ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
# print("ARD by images: ", ard)
# print("Average ARD: ", ard.mean())
# plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Clark Error Grid Zone: ", zone)
# risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Surveillance Error Grid Zone: ", risk_levels_percentage)

# # PLS
# print("PLS")
# model = PLSRegression()
# model.fit(X_train, y_train)
# y_pred_pls = model.predict(X_test)
# ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
# print("ARD by images: ", ard)
# print("Average ARD: ", ard.mean())
# plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Clark Error Grid Zone: ", zone)
# risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Surveillance Error Grid Zone: ", risk_levels_percentage)

# # SVM
# print("SVM")
# model = cuSVR()
# model.fit(X_train, y_train)
# y_pred_svm = cp.asnumpy(model.predict(X_test))
# ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
# print("ARD by images: ", ard)
# print("Average ARD: ", ard.mean())
# plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Clark Error Grid Zone: ", zone)
# risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Surveillance Error Grid Zone: ", risk_levels_percentage)

# # Linear SVR
# print("Linear SVR")
# model = LinearSVR(random_state=42)
# model.fit(X_train, y_train)
# y_pred_svr = cp.asnumpy(model.predict(X_test))
# ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
# print("ARD by images: ", ard)
# print("Average ARD: ", ard.mean())
# plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Clark Error Grid Zone: ", zone)
# risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Surveillance Error Grid Zone: ", risk_levels_percentage)

# # SVR
# print("SVR")
# model = SVR()
# print("Start fitting model...")
# model.fit(X_train, y_train)
# print("End fitting model...")
# y_pred_svr = cp.asnumpy(model.predict(X_test))
# ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
# # print("ARD by images: ", ard)
# print("Average ARD: ", ard.mean())
# plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Clark Error Grid Zone: ", zone)
# risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False)
# print("Surveillance Error Grid Zone: ", risk_levels_percentage)

logging.info('Done')
