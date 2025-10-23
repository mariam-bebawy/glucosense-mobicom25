import os, sys
import numpy as np
import pickle
from tqdm import tqdm
import cupy as cp

import cuml
from cuml.svm import SVR as SVR_GPU
from cuml.ensemble import RandomForestRegressor as RF_GPU
from cuml import LinearRegression as LR_GPU
from cuml import Lasso as LAS_GPU
from cuml import Ridge as RR_GPU
from cuml import ElasticNet as EN_GPU

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import ParameterSampler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


class Regressors(object):


    def __init__(self, output_path):
        self.output_path = output_path
        self.parameters = None
        self.n_jobs = -1
        self.random_state = 42
      

    def model_search(self, classifier, X_train, y_train, X_val, y_val, X_test, y_test, gpu=False):

        print(f"Performing cross validation using {classifier}...")
        if classifier == "RF":
            from hyperparam_config import rgr_random_forest_params as hyperparams
        elif classifier == "LR":
            from hyperparam_config import rgr_linear_regression_params as hyperparams
        elif classifier == "MLP":
            from hyperparam_config import rgr_mlp_params as hyperparams
        elif classifier == "SVR":
            from hyperparam_config import rgr_svm_params as hyperparams
        elif classifier == "PLS":
            from hyperparam_config import rgr_pls_params as hyperparams
        elif classifier == "XGB":
            from hyperparam_config import rgr_xgb_params as hyperparams
        elif classifier == "RR":
            from hyperparam_config import rgr_rr_params as hyperparams
        elif classifier == "LAS":
            from hyperparam_config import rgr_las_params as hyperparams
        elif classifier == "EN":
            from hyperparam_config import rgr_en_params as hyperparams
        else:
            raise Exception("Undefined classifier! Please provide a valid classifier name")

        param_list = list(ParameterSampler(hyperparams, n_iter=800))
        # append default params
        param_list.append({})
        
        param_dict_result = dict()
        best_result_param = ""
        best_r2 = -np.inf
        best_model = None
        best_results_list = []
        final_preds_labels = []
        print("Starting hyperparameter search...")
        for param in tqdm(param_list):
            param_str = self.dict_to_str(param)
            param_dict_result[param_str] = dict()

            regressor = None

            if gpu:
                X_train = cp.array(X_train)
                y_train = cp.array(y_train)
                X_val = cp.array(X_val)
                y_val = cp.array(y_val)
                X_test = cp.array(X_test)
                y_test = cp.array(y_test)

                if classifier == "SVR":
                    regressor = SVR_GPU(**param)
                elif classifier == "RF":
                    regressor = RF_GPU(**param)
                elif classifier == "LR":
                    regressor = LR_GPU(**param)
                elif classifier == "MLP":
                    regressor = MLPRegressor(**param)
                elif classifier == "PLS":
                    regressor = PLSRegression(**param)
                elif classifier == "XGB":
                    regressor = XGBRegressor(**param, device="cuda", tree_method="hist", n_jobs=self.n_jobs)
                elif classifier == "RR":
                    regressor = RR_GPU(**param)
                elif classifier == "LAS":
                    regressor = LAS_GPU(**param)
                elif classifier == "EN":
                    regressor = EN_GPU(**param)
            
            else:
                if classifier == "SVR":
                    regressor = SVR(**param)
                elif classifier == "RF":
                    regressor = RandomForestRegressor(n_jobs=self.n_jobs, **param)
                elif classifier == "LR":
                    regressor = LinearRegression(n_jobs=self.n_jobs, **param)
                elif classifier == "MLP":
                    regressor = MLPRegressor(**param)
                elif classifier == "PLS":
                    regressor = PLSRegression(**param)
                elif classifier == "XGB":
                    regressor = XGBRegressor(**param)
            
            model = regressor.fit(X_train, y_train)
            val_predictions = model.predict(X_val)
            test_predictions = model.predict(X_test)
            if gpu:
                val_predictions = cp.asnumpy(val_predictions)
                test_predictions = cp.asnumpy(test_predictions)
                y_val = cp.asnumpy(y_val)
                y_test = cp.asnumpy(y_test)
            val_r2_score = r2_score(y_val, val_predictions)
            val_mse_score = mean_squared_error(y_val, val_predictions)
            test_r2_score = r2_score(y_test, test_predictions)
            test_mse_score = mean_squared_error(y_test, test_predictions)
            result_list = [val_r2_score, val_mse_score, test_r2_score, test_mse_score]
            result_list = [round(val, 3) for val in result_list]
            param_dict_result[param_str] = result_list
            
            if val_r2_score > best_r2:
                best_r2 = val_r2_score
                best_result_param = param_str
                best_model = model
                best_results_list = result_list


        print("Hyperparameter search finished...")
        
        print("Saving the best model and outputs ...")
        if self.output_path is not None:
            with open(self.output_path+"_performance_results.txt", "w") as f:
                f.write("Hyperparameters: "+best_result_param)
                f.write("\t".join(["val_r2_score", "val_mse_score", "test_r2_score", "test_mse_score"])+"\n")
                f.write("\t".join([str(val) for val in best_results_list]))

            with open(self.output_path+".model", 'wb') as f:
                pickle.dump(best_model,f)

            
        print("Best ave. r2", best_r2, "best param", best_result_param)
        return best_model
        
    def dict_to_str(self, i_dict):
        # an empty string
        converted = str()
        for key in i_dict:
            converted += key + ": " + str(i_dict[key]) + ", "
        return converted