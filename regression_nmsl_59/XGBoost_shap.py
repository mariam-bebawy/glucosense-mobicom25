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

BANDS_WAVELENGTHS = [397.32, 400.20, 403.09, 405.97, 408.85, 411.74, 414.63, 417.52, 420.40, 423.29, 426.19, 429.08, 431.97, 434.87, 437.76, 440.66, 443.56, 446.45, 449.35, 452.25, 455.16, 458.06, 460.96, 463.87, 466.77, 469.68, 472.59, 475.50, 478.41, 481.32, 484.23, 487.14, 490.06, 492.97, 495.89, 498.80, 501.72, 504.64, 507.56, 510.48, 513.40, 516.33, 519.25, 522.18, 525.10, 528.03, 530.96, 533.89, 536.82, 539.75, 542.68, 545.62, 548.55, 551.49, 554.43, 557.36, 560.30, 563.24, 566.18, 569.12, 572.07, 575.01, 577.96, 580.90, 583.85, 586.80, 589.75, 592.70, 595.65, 598.60, 601.55, 604.51, 607.46, 610.42, 613.38, 616.34, 619.30, 622.26, 625.22, 628.18, 631.15, 634.11, 637.08, 640.04, 643.01, 645.98, 648.95, 651.92, 654.89, 657.87, 660.84, 663.81, 666.79, 669.77, 672.75, 675.73, 678.71, 681.69, 684.67, 687.65, 690.64, 693.62, 696.61, 699.60, 702.58, 705.57, 708.57, 711.56, 714.55, 717.54, 720.54, 723.53, 726.53, 729.53, 732.53, 735.53, 738.53, 741.53, 744.53, 747.54, 750.54, 753.55, 756.56, 759.56, 762.57, 765.58, 768.60, 771.61, 774.62, 777.64, 780.65, 783.67, 786.68, 789.70, 792.72, 795.74, 798.77, 801.79, 804.81, 807.84, 810.86, 813.89, 816.92, 819.95, 822.98, 826.01, 829.04, 832.07, 835.11, 838.14, 841.18, 844.22, 847.25, 850.29, 853.33, 856.37, 859.42, 862.46, 865.50, 868.55, 871.60, 874.64, 877.69, 880.74, 883.79, 886.84, 889.90, 892.95, 896.01, 899.06, 902.12, 905.18, 908.24, 911.30, 914.36, 917.42, 920.48, 923.55, 926.61, 929.68, 932.74, 935.81, 938.88, 941.95, 945.02, 948.10, 951.17, 954.24, 957.32, 960.40, 963.47, 966.55, 969.63, 972.71, 975.79, 978.88, 981.96, 985.05, 988.13, 991.22, 994.31, 997.40, 1000.49, 1003.58]
num_features_to_select = 50


data_root = '../../datasets/HSDatasets/'
file_train = 'hs_train.csv'
file_test = 'hs_test.csv'
step = 1

X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, step, 'hyper')
# X_train, X_val, y_train, y_val = TTS(X_train, y_train, test_size=0.3, random_state=42)

# Scale data
# scaler = StandardScaler()
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#model = XGBRegressor(random_state=42, n_jobs=-1, device='cuda', tree_method='hist')
model = XGBRegressor(n_estimators=800
                   ,subsample=0.894736842105263
                   ,random_state=42
                   ,tree_method='hist'
                   ,device='cuda'
                   ,n_job=-1
                   ,learning_rate=0.26
                   ,gamma=0.006
                   ,max_depth=8)

# Shap values
shap_model = model
shap_model.fit(X_train, y_train)

ard = ard_image(test_data, shap_model, scaler_X=scaler)
print("ARD:::",ard.mean())
# # Save the model
# pickle.dump(shap_model, open('./Models/XGB_default.pkl', 'wb'))

# Create a SHAP explainer
explainer = shap.TreeExplainer(shap_model, X_train)
# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Create and save the SHAP summary plot
#shap.plots.beeswarm(shap_values)
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS[1:204:step], max_display=204, show=False)
plt.savefig('XGB_shap_builtin_summary_plot.png', bbox_inches='tight')
plt.close()  # Close the plot to avoid displaying it

# # Random Forest
# # The summary plot gives a global overview of the feature importance 
# # and the effects of the features on the model's predictions.
# shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=50)

# # Random Forest
# # The summary plot gives a global overview of the feature importance 
# # and the effects of the features on the model's predictions.
# shap.summary_plot(shap_values, X_test, feature_names=BANDS_WAVELENGTHS, max_display=50, plot_type='bar')


# Calculate mean absolute SHAP values for each feature
mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)

# Plot using matplotlib
plt.figure(figsize=(20, 10))
plt.bar(range(len(BANDS_WAVELENGTHS[1:204:step])), mean_abs_shap_values)
plt.ylabel('Mean |SHAP Value|')
plt.xlabel('Feature')
plt.title('Feature Importance based on Mean Absolute SHAP Values')
plt.xticks(range(0, 204, 10), BANDS_WAVELENGTHS[0::10], rotation=90)

# Save the plot to a file
plt.savefig('XGB_shap_summary_plot.png', bbox_inches='tight')
