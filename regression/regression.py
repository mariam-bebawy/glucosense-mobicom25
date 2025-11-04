import copy
import numpy as np
import cuml
import cupy as cp
import scipy
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
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
import matplotlib.pyplot as plt
import os
os.chdir("/local-scratch/GlucoseProject/mobicom23_mobispectral/regression")
import sys
sys.path.append(os.getcwd())
from utils import evaluate_image, evaluate_image_SEG, prepare_data, ard_image, make_train_test_files, save_model, load_model
from evaluate import ARD, CDF, clarke_error_grid, plot_surveillance_error_grid, calculate_seg_risks, plot_overlapped_error_grids 

from Architecture.MLP import MLPRegressorPyTorch

BANDS_WAVELENGTHS = [397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68, 932.74, 935.81, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58]

# %%
#data_root = '../../datasets/MobileDatasets/RaspberryPi-RGB+ToF'
#data_root = '../../datasets/MobileDatasets/GooglePixel4XL'
#data_root = '../../datasets/MobileDatasets/OnSemi-custom-RGB-NIR'
data_root = '../../datasets/MobileDatasets/RaspberryPi-RGB-NoIR'
#data_root = '../../datasets/HSDatasets'
#data_root = '../../datasets/working_glucose/Hyperspectral'
#data_root = '../../datasets/working_glucose/mobile/GooglePixel4XL'
#data_root = '../../datasets/working_glucose/mobile/RaspberryPi-RGB+ToF'
#data_root = '../../datasets/working_glucose/mobile/OnSemi-custom-RGB-NIR'
file_train = 'PiNoIR_train.csv'
file_test = 'PiNoIR_test.csv'
step = 1


X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, step, 'mobi')


#X = np.concatenate((X_train, X_test), axis=0)
#y = np.concatenate((y_train, y_test), axis=0)
#print(X.shape, y.shape)

#mi = mutual_info_regression(X, y)

# Plot mutual information
#plt.figure(figsize=(20, 10))
#plt.bar(range(len(BANDS_WAVELENGTHS)), mi)
#plt.xlabel('Wavelength (nm)')
#plt.ylabel('Mutual Information')
#plt.title('Mutual Information of selected bands')
# xticks is band wavelengths, but 10 apart from each tick
#plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)
#plt.show()

# %%
#scaler = MinMaxScaler()
scaler_train = RobustScaler()
# scaler = StandardScaler()
X_train = scaler_train.fit_transform(X_train)
# X_val = scaler.transform(X_val)
X_test = scaler_train.transform(X_test)
scaler_test = RobustScaler()
y_train = scaler_test.fit_transform(y_train.reshape(-1,1))
y_test = scaler_test.transform(y_test.reshape(-1,1))


# %%
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# %%
# for image in test_data:
#     image['sig'] = scaler.transform(image['sig'])

#model = MLPRegressorPyTorch((200, 100, 16), random_state=42, batch_size=50000, learning_rate_init=0.001, max_iter=200, verbose=True)
#model = MLPRegressor(hidden_layer_sizes=(200,100,16), random_state=42 ,max_iter=200, activation='relu', solver='adam', alpha=0.0001)
#model =  XGBRegressor(device='cuda', tree_method='hist')
model = XGBRegressor(n_estimators=800
                   ,subsample=0.894736842105263
                   ,random_state=42
                   ,tree_method='hist'
                   ,device='cuda'
                   ,n_job=-1
                   ,learning_rate=0.26
                   ,gamma=0.006
                   ,max_depth=8).fit(X_train,y_train)
#model.fit(X_train, y_train, X_test, y_test)
model.fit(X_train, y_train)
#save_model(model, './Models/final/xgboost_PiNoIR.pth')
#model = load_model('./Models/xgboost_hs_68bands.pth')

#with torch.no_grad():

   # # Save the trained model
   # torch.save(model.state_dict(), './Models/FFNN_test4.0.pth')


   # # Load the trained model
   # model.load_state_dict(torch.load("./Models/FFNN_test_ToF1.0.pth"))

predictions = []
labels = []

predictions = np.asarray(predictions)
labels = np.asarray(labels)

# X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

# for i in range(len(test_data)):
#     test_img = test_data[i]
#     X_test_scaled = scaler_X.transform(test_img['sig'])
#     X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
#     y_pred = model(X_test_scaled).cpu().numpy()
#     # y_pred need to retransform to the original scale
#     y_pred = scaler_y.inverse_transform(y_pred)
#     y_pred = y_pred.mean()
#     predictions = np.append(predictions, y_pred)
#     labels = np.append(labels, test_img['label'][1])

for i in range(len(test_data)):
    test_img = test_data[i]
    X_test = test_img['sig']
    X_test = scaler_train.transform(X_test)
    y_pred = model.predict(X_test)
    y_pred_mean = y_pred.mean()
    y_pred_mean = scaler_test.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
    predictions = np.append(predictions, y_pred_mean)
    labels = np.append(labels, test_img['label'][1])

    # y_pred = model(X_test).cpu().numpy()
    # print(f"ARD by signatures: {ARD(y_pred, y_test).mean()}" )
    # evaluate_image(test_data, model=model)

print("Pred:",predictions)
print("Labels",labels)
ard = ARD(predictions, labels)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())

r, p = scipy.stats.pearsonr(predictions, labels)
print("PCC:r ", r)
print("PCC:p ", p)

plot_overlapped_error_grids(predictions, labels, './graph/SEG/seg600.png')
#plt2, zones = clarke_error_grid(predictions, labels, 'Clark')
#plt2.show()
#print("Clark zones:",zones)
#plot_surveillance_error_grid(predictions, labels, './graph/SEG/seg600.png')
#calculate_seg_risks(predictions, labels, 3)
#CDF(ard)















'''# Multi-Layer Perceptron
mlp_reg = MLPRegressor(hidden_layer_sizes=(200,200), random_state=42 ,max_iter=200, activation='relu', solver='adam', alpha=0.0001)
mlp_reg.fit(cp.asnumpy(X_train), cp.asnumpy(y_train))
y_pred_mlp = cp.asnumpy(mlp_reg.predict(X_test))
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
#print(f"ARD by signatures: {ARD(y_pred_mlp, y_test).mean()}" )
evaluate_image(test_data, model=mlp_reg, scaler_X=scaler_train, scaler_y=scaler_test)
ard = ard_image(test_data, mlp_reg, scaler_X=scaler_train, scaler_y=scaler_test)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())
evaluate_image_SEG(test_data, model=mlp_reg, scaler_X=scaler_train, scaler_y=scaler_test)
CDF(ard)

# XGBoost
xgb_reg = XGBRegressor(device='cuda', tree_method='hist')
xgb_reg.fit(X_train, y_train)
y_pred_xgb = cp.asnumpy(xgb_reg.predict(X_test))
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"ARD by signatures: {ARD(y_pred_xgb, y_test).mean()}" )
evaluate_image(test_data, model=xgb_reg, scaler_X=scaler_train, scaler_y=scaler_test)
ard = ard_image(test_data, xgb_reg, scaler_X=scaler_train, scaler_y=scaler_test)
print("ARD by images: ", ard)
print("Average ARD: ", ard.mean())
evaluate_image_SEG(test_data, model=xgb_reg, scaler_X=scaler_train, scaler_y=scaler_test)
CDF(ard)
'''
