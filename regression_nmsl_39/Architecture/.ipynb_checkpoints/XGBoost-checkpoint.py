import pickle

from scipy.stats import uniform, randint

import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import argparse
import time
import logging
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

from mobicom23_mobispectral.regression.evaluate import *
from mobicom23_mobispectral.regression.Architecture.utils import *
from mobicom23_mobispectral.regression.dict_dataset import ImageList
from scipy.stats import pearsonr



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')


def fit_model(model, X_train, X_test, y_train, y_test, model_name):
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logging.info("\n{}:".format(model_name))
    logging.info(confusion_matrix(y_test, y_pred))
    logging.info(classification_report(y_test, y_pred))
    logging.info("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
    logging.info("Time taken: {}".format(time.time() - start))


def plot_losses(model):
    plt.plot(model.loss_curve_)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()


def prepare_data(data_root, file_train, file_test, step):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    image_list1 = ImageList(data_root, file_train, step)
    for i in range(len(image_list1)):
        image = image_list1[i]
        for sig in image['sig']:
            X_train.append(sig)
        for label in image['label']:
            y_train.append(label)

    image_list2 = ImageList(data_root, file_test, step)
    # for i in range(len(image_list2)):
    #     image = image_list2[i]
    #     for sig in image['sig']:
    #         X_test.append((sig, i))
    #     for label in image['label']:
    #         y_test.append((label, i))


    # train_data = TrainDataset(data_root, file_train, step)
    # test_data = TestDataset(data_root, file_test, step)

    # for label, sig in train_data:
    #     X_train.append(sig)
    #     #plt.plot(sig)
    #     #plt.show()
    #     y_train.append(label)
    #
    # for label, sig in test_data:
    #     # min = sig.min()
    #     # max = sig.max()
    #     # sig = (sig - min) / (max - min)
    #     X_test.append(sig)
    #     y_test.append(label)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    # X_test = np.asarray(X_test)
    # y_test = np.asarray(y_test)

    logging.info('Shape of X_train: {}'.format(X_train.shape))
    logging.info('Shape of y_train: {}'.format(y_train.shape))
    # logging.info('Shape of X_test: {}'.format(X_test.shape))
    # logging.info('Shape of y_test: {}'.format(y_test.shape))

    return X_train, y_train, image_list2


def configure_and_train_xgb(X_train, y_train):
    param_dist = {
        'colsample_bytree': uniform(0.7, 0.3),  # Uniform distribution between 0.7 and 1.0
        'learning_rate': uniform(0.01, 0.1),  # Uniform distribution between 0.01 and 0.11
        'max_depth': randint(3, 8),  # Discrete uniform distribution between 3 and 7
        'alpha': randint(5, 16),  # Discrete uniform distribution between 5 and 15
        'n_estimators': randint(50, 151)  # Discrete uniform distribution between 50 and 150
    }

    # Initialize the model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    #
    # # Setup the random search with cross-validation
    # random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist,
    #                                    n_iter=100, cv=3, scoring='neg_mean_squared_error', verbose=1,
    #                                    random_state=42)
    #
    # # Perform the random search
    # random_search.fit(X_train, y_train)
    #
    # # Output the best parameters and the best model
    # logging.info("Best parameters found: ", random_search.best_params_)
    # best_model = random_search.best_estimator_

    # filename = '../Models/xgb_1.pki'
    # with open(filename, 'rb') as file:
    #     loaded_model = pickle.load(file)
    # best_model = loaded_model

    return xgb_model


def evaluate_clarke_error_grid(y_test, y_pred, show=False):
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
    try:
        y_test = y_test.ravel()
        y_pred = y_pred.ravel()
        ard = ARD(y_pred, y_test)
        logging.info('ARD: {}'.format(ard))
        ard_accurate = np.sum(ard < 0.2)
        percentage_ard_accurate = ard_accurate * 100 / len(ard)
        logging.info(
            'Percentage of clinical accurate glucose concentration (ARD < 0.2): {}/{} = {:.2f}%'.format(ard_accurate,
                                                                                                        len(ard),
                                                                                                        percentage_ard_accurate))
        if show:
            show_ARD(ard)
        return ard
    except Exception as e:
        logging.error("Failed to calculate ARD: {}".format(str(e)))


def evaluate_pearson_correlation(y_pred, y_test):
    try:
        y_test = y_test.ravel()
        y_pred = y_pred.ravel()
        corr, _ = pearsonr(y_pred, y_test)
        logging.info('Pearson correlation coefficient: {:.3f}'.format(corr))

        return corr
    except Exception as e:
        logging.error("Failed to calculate Pearson correlation: {}".format(str(e)))


def main_evaluation(y_pred, y_test):
    zone = evaluate_clarke_error_grid(y_test, y_pred, show=True)
    ard = evaluate_ard(y_pred, y_test, show=False)
    corr, pv = pearsonr(y_pred, y_test)
    return ard, corr, pv


def main(data_root, file_train, file_test):
    ARD = []
    ARD = np.asarray(ARD)
    Pearson = []
    Pearson = np.asarray(Pearson)

    for step in range(1, 11):
        X_train, y_train, test_data = prepare_data(data_root, file_train, file_test, step)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        corr_f, p_f = select_feature_PCC(X_train, y_train)
        show_correlation_P(corr_f, p_f)
        #X_test_scaled = scaler.transform(X_test[:, 0]) # X_test shape: (sig, image_index)
        xgb_model = configure_and_train_xgb(X_train_scaled, y_train)
        # save_model(xgb_model, f'../Models/xgb_{step}.pkl')
        predictions = []
        labels = []

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        for i in range(len(test_data)):
            test_img = test_data[i]
            X_test_scaled = scaler.transform(test_img['sig'])
            y_pred = xgb_model.predict(X_test_scaled)
            y_pred = y_pred.mean()
            predictions = np.append(predictions, y_pred)
            labels = np.append(labels, test_img['label'][1])
        # plt2, _ = clarke_error_grid(labels, predictions, 'Clark')
        # plt2.show()



        #pred_image = get_image_prediction(y_pred, y_test)
        #test_image = get_image_reference(y_test, y_test)

        ard, corr, pv = main_evaluation(predictions, labels)
        ard = ard.mean()
        ARD = np.append(ARD, ard)
        Pearson = np.append(Pearson, corr)
        logging.info('Mean ARD of the step {} is {}'.format(step, ard))
    plt.figure()
    plt.plot(range(1,11),ARD)
    plt.title('ARD performance varies on number of bands')
    plt.xlabel('Step size')
    plt.ylabel('ARD')
    plt.show()
    plt.figure()
    plt.plot(range(1, 11), Pearson)
    plt.title('Pearson performance varies on number of bands')
    plt.xlabel('Step size')
    plt.ylabel('PCC')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Regression Toolbox")
    parser.add_argument('--data_root', type=str, default='../../datasets/dataset_skin/regression')
    parser.add_argument('--file_train', type=str, default='concatset_train.txt')
    parser.add_argument('--file_test', type=str, default='concatset_test.txt')
    opt = parser.parse_args()

    main(opt.data_root, opt.file_train, opt.file_test)
