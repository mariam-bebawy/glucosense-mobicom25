import pickle
import numpy as np
import cupy as cp
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import argparse
import time
import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import os
os.chdir("/local-scratch/GlucoseProject/mobicom23_mobispectral/regression")
import sys
sys.path.append(os.getcwd())
from evaluate import *
from utils import *
from dict_dataset import ImageList
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fit_model(model, X_train, X_test, y_train, y_test, model_name):
    """Fit the model and evaluate its performance."""
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info("\n{}:".format(model_name))
    logging.info(confusion_matrix(y_test, y_pred))
    logging.info(classification_report(y_test, y_pred))
    logging.info("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    logging.info("Time taken: {}".format(time.time() - start))



def configure_and_train_xgb(X_train, y_train):
    """Configure and train the XGBoost model using GridSearchCV."""

    param = {
        'tree_method': 'hist',
        'device': 'cuda'
    }


    param_grid = {
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, 9, 11],
        'reg_alpha': [0, 1, 5, 10, 15, 20],
        'n_estimators': [50, 100, 150, 200, 300]
    }

    xgb_model = XGBRegressor(objective='reg:squarederror', **param)

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               cv=3, verbose=1, n_jobs=-1)

    # Check if XGBoost is using GPU
    training_params = xgb_model.get_xgb_params()
    logging.info("Training device: %s", training_params['tree_method'])

    grid_search.fit(X_train, y_train)

    logging.info("Best parameters found: %s", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    filename = '../Models/xgb_test_band68.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(best_model, file)

    return best_model

def evaluate_clarke_error_grid(y_test, y_pred, show=False):
    """Evaluate using Clarke Error Grid analysis."""
    try:
        plt.figure()
        plt2, zone = clarke_error_grid(y_test, y_pred,
                                       'Clinical Standard Error Grid for Glucose Reference and Predicted Value')
        logging.info("Clarke Error Grid analysis completed.")
        if show:
            plt2.show()
        return zone
    except Exception as e:
        logging.error("Failed to generate Clarke Error Grid: {}".format(str(e)))

def evaluate_ard(y_pred, y_test, show=False):
    """Evaluate using ARD (Absolute Relative Difference)."""
    try:
        y_test, y_pred = y_test.ravel(), y_pred.ravel()
        ard = ARD(y_pred, y_test)
        logging.info('ARD: {}'.format(ard))
        ard_accurate = np.sum(ard < 0.2)
        percentage_ard_accurate = ard_accurate * 100 / len(ard)
        logging.info('Percentage of clinically accurate glucose concentration (ARD < 0.2): {}/{} = {:.2f}%'.format(
            ard_accurate, len(ard), percentage_ard_accurate))
        if show:
            show_ARD(ard)
        return ard
    except Exception as e:
        logging.error("Failed to calculate ARD: {}".format(str(e)))

def evaluate_pearson_correlation(y_pred, y_test):
    """Evaluate using Pearson correlation coefficient."""
    try:
        y_test, y_pred = y_test.ravel(), y_pred.ravel()
        corr, _ = pearsonr(y_pred, y_test)
        logging.info('Pearson correlation coefficient: {:.3f}'.format(corr))
        return corr
    except Exception as e:
        logging.error("Failed to calculate Pearson correlation: {}".format(str(e)))

def main_evaluation(y_pred, y_test):
    """Run main evaluation metrics."""
    zone = evaluate_clarke_error_grid(y_test, y_pred, show=True)
    ard = evaluate_ard(y_pred, y_test, show=False)
    corr, pv = pearsonr(y_pred, y_test)
    return ard, corr, pv

def main(data_root, file_train, file_test):
    """Main function to prepare data, train model, and evaluate."""
    ARD, Pearson = [], []
    step = 3
    X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, file_train, file_test, step)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = cp.array(X_train_scaled)
    y_train = cp.array(y_train)
    xgb_model = configure_and_train_xgb(X_train_scaled, y_train)
    save_model(xgb_model, '../Models/xgb_band68.pkl')
    # param = {
    #     'max_depth':5,
    #     'subsample':0.8, 
    #     'colsample_bytree':0.8,
    #     'eta':0.5,
    #     'min_child_weight':1,
    #     'tree_method':'gpu_hist',
    #     'objective': 'reg:squarederror',
    #     'device': 'cuda'
    # }
    # num_round = 100

    # dtrain = xgb.DMatrix(X_train_scaled, y_train)
    # xgb_model = xgb.train(param, dtrain, num_round)

    # xgb_model = XGBRegressor(**param)
    # xgb_model.fit(X_train_scaled, y_train)


    predictions, labels = [], []
    
    for test_img in test_data:
        X_test_scaled = scaler.transform(test_img['sig'])
        y_pred = xgb_model.predict(X_test_scaled)
        y_pred_mean = y_pred.mean()
        predictions.append(y_pred_mean)
        labels.append(test_img['label'][1])
    
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    ard, corr, pv = main_evaluation(predictions, labels)
    ARD.append(ard.mean())
    Pearson.append(corr)
    logging.info('Mean ARD of the step {} is {}'.format(step, ard.mean()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Regression Toolbox")
    parser.add_argument('--data_root', type=str, default='../datasets/dataset_skin/regression')
    parser.add_argument('--file_train', type=str, default='labels_s1_train.txt')
    parser.add_argument('--file_test', type=str, default='labels_s1_test.txt')
    opt = parser.parse_args()

    main(opt.data_root, opt.file_train, opt.file_test)
