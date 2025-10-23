import pickle

import numpy as np
import pandas as pd
import cupy as cp
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import argparse
import os
import logging
import hdf5storage
import csv

os.chdir("/local-scratch/GlucoseProject/mobicom23_mobispectral/regression")
from dict_dataset import ImageList
from evaluate import *
#from config import *

# parser = argparse.ArgumentParser(description="Spectral Regression Toolbox")
# parser.add_argument('--data_root', type=str, default='../../datasets/dataset_skin/regression')
# opt = parser.parse_args()

def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def prepare_data(data_root, file_train, file_test, step, dataset):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    image_list1 = ImageList(data_root, file_train, step, dataset)
    for i in range(len(image_list1)):
        image = image_list1[i]
        for sig in image['sig']:
            # plt.plot(sig)
            X_train.append(sig)
        for label in image['label']:
            y_train.append(label)

    image_list2 = ImageList(data_root, file_test, step, dataset)
    for i in range(len(image_list2)):
        image = image_list2[i]
        for sig in image['sig']:
            X_test.append(sig)
        for label in image['label']:
            y_test.append(label)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    logging.info('Shape of X_train: {}'.format(X_train.shape))
    logging.info('Shape of y_train: {}'.format(y_train.shape))
    logging.info('Shape of X_test: {}'.format(X_test.shape))
    logging.info('Shape of y_test: {}'.format(y_test.shape))

    return X_train, y_train, X_test, y_test, image_list2

def prepare_data2(data_root, file_name, step):
    X = []
    y = []

    image_list = ImageList(data_root, file_name, step)
    for i in range(len(image_list)):
        image = image_list[i]
        for sig in image['sig']:
            X.append(sig)
        for label in image['label']:
            y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)

    logging.info('Shape of X: {}'.format(X.shape))
    logging.info('Shape of y: {}'.format(y.shape))

    return X, y, image_list

# Get all training and testing data, and return the corresponding image_list
def get_all_data(data_root, file_train, file_test, step, dataset):
    X = []
    y = []

    image_list1 = ImageList(data_root, file_train, step, dataset)
    for i in range(len(image_list1)):
        image = image_list1[i]
        for sig in image['sig']:
            plt.plot(sig)
            X.append(sig)
        for label in image['label']:
            y.append(label)

    image_list2 = ImageList(data_root, file_test, step, dataset)
    for i in range(len(image_list2)):
        image = image_list2[i]
        for sig in image['sig']:
            X.append(sig)
        for label in image['label']:
            y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)

    logging.info('Shape of X: {}'.format(X.shape))
    logging.info('Shape of y: {}'.format(y.shape))

    return X, y, image_list1, image_list2


def select_feature_PCC(X, y):
    # Flatten y to a 1D array to ensure compatibility with pearsonr
    y = y.ravel()
    correlations = []
    p = []
    for band in range(X.shape[1]):
        if np.unique(X[:, band]).size > 1:  # Prevent zero
            corr, pv = pearsonr(X[:, band], y)
            correlations.append(corr)
            p.append(pv)
        else:
            correlations.append(np.nan)  # Append NaN if data is invalid
    return correlations, p

def show_correlation_P(correlations, p):
    fig, axs = plt.subplots(1, 2)  # 1 row, 2 columns

    # First plot
    axs[0].plot(correlations, 'r-')
    axs[0].set_title('PCC over bands')
    axs[0].set_xlabel('Bands')
    axs[0].set_ylabel('PCC')

    # Second plot
    axs[1].plot(p, 'b-')
    axs[1].set_title('P-value over bands')
    axs[1].set_xlabel('Bands')
    axs[1].set_ylabel('Probability')

    plt.tight_layout()
    plt.show()

def show_ARD(ard):
    plt.plot(ard)
    plt.title('Absolute relative difference between predicted and reference glucose reading')
    plt.xlabel('Image index') # Later may change to Image
    plt.ylabel('ARD')
    plt.show()

def show_dataset_heatmap(X):
    plt.imshow(X, aspect='auto', cmap='inferno')  # 'inferno' is just one example of a colormap
    plt.colorbar()  # Shows the color scale
    plt.title('Heatmap of Signatures by Bands')
    plt.xlabel('Bands')
    plt.ylabel('Signatures')
    plt.show()

def show_dataset_multiplot(X):
    for sig in range(X.shape[0]):
        max = X[sig, :].max()
        min = X[sig, :].min()
        X[sig, :] = (X[sig, :] - min)/(max - min)
        plt.plot(np.squeeze(X[sig, :]))
    plt.show()

def get_image_prediction(y_pred, y_test):
    ## Initialize variables
    current_index = y_test[0][1]
    value_sum = 0
    count = 0
    averages = {}

    # Iterate through the data
    for i, value, index in y_test:
        if index == current_index:
            # Not the value of y_test
            value = y_pred[i]
            value_sum += value
            count += 1
        else:
            # Calculate average for the current index
            averages[current_index] = value_sum / count
            # Reset for the next index
            current_index = index
            value_sum = value
            count = 1

    # Calculate the average for the last group
    if count > 0:
        averages[current_index] = value_sum / count

    print("Averages by index:")
    for index, avg in averages.items():
        print(f"Index {index}: {avg}")

    return averages

def get_image_reference(y_test):
    ## Initialize variables
    current_index = y_test[0][1]
    value_sum = 0
    count = 0
    averages = {}

    # Iterate through the data
    for i, value, index in y_test:
        if index == current_index:
            value_sum += value
            count += 1
        else:
            # Calculate average for the current index
            averages[current_index] = value_sum / count
            # Reset for the next index
            current_index = index
            value_sum = value
            count = 1

    # Calculate the average for the last group
    if count > 0:
        averages[current_index] = value_sum / count

    print("Averages by index:")
    for index, avg in averages.items():
        print(f"Index {index}: {avg}")

    return averages

def ard_image(test_data, model=None, path=None, scaler_X=None, scaler_y=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    ard = []

    ard = np.asarray(ard, dtype=np.float32)

    for i in range(len(test_data)):
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        if scaler_X is not None:
            X_test = scaler_X.transform(X_test)
        y_pred = model.predict(X_test)
        y_pred = cp.asnumpy(y_pred)
        y_pred_mean = y_pred.mean()
        if scaler_y is not None:
            y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
        ard = np.append(ard, ARD(y_pred_mean, test_img['label'][0]))
    return ard

def evaluate_image(test_data, model=None, path=None, scaler_X=None, scaler_y=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)


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
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        if scaler_X is not None:
            X_test = scaler_X.transform(X_test)
        y_pred = model.predict(X_test)
        y_pred = cp.asnumpy(y_pred)
        y_pred_mean = y_pred.mean()
        if scaler_y is not None:
            y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, test_img['label'][0])

    plt2, _ = clarke_error_grid(predictions, labels, 'Clark')
    plt2.show()

def evaluate_image_SEG(test_data, model=None, path=None, scaler_X=None, scaler_y=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)


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
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        if scaler_X is not None:
            X_test = scaler_X.transform(X_test)
        y_pred = model.predict(X_test)
        y_pred = cp.asnumpy(y_pred)
        y_pred_mean = y_pred.mean()
        if scaler_y is not None:
            y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, test_img['label'][0])

    plot_surveillance_error_grid(predictions, labels, 'graph/SEG/seg600.png')

def evaluate_risk_SEG(test_data, model=None, path=None, patient_id=None, scaler_X=None, scaler_y=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)


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
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        if scaler_X is not None:
            X_test = scaler_X.transform(X_test)
        y_pred = model.predict(X_test)
        y_pred = cp.asnumpy(y_pred)
        y_pred_mean = y_pred.mean()
        if scaler_y is not None:
            y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, test_img['label'][0])

    calculate_seg_risks(predictions, labels, patient_id)


def show_image(file, bands):
        # hyper_data_path = f'{data_root}/HS_GT/'
        # bgr_data_path = f'{data_root}/RGBN/'
        # test_data_path = f'{data_root}/testing/'
        # mask_data_path = f'{data_root}/masks/'
        # with open(f'{data_root}/{file}', 'r') as fin:
        #     # Skip the first line
        #     fin.readline()
        #     lines = [line.split(',') for line in fin]
        #     # hyper_list = [l[0] + '.mat' for l in lines]
        #     hyper_list = [l[1] + '.mat' for l in lines]
        #     # print(hyper_list)
        #     # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
        #     label_list = [np.float32(l[2].replace('\n', '')) for l in lines]
        #     # print(label_list)
        #     # bgr_list = [line.replace('.mat','_RGB.png') for line in hyper_list]
        #     mask_list = [line.replace('.mat', '.png') for line in hyper_list]
        # logging.info(f'len(hyper) of MobiSpectral dataset:{len(hyper_list)}')
        # logging.info(f'len(label) of MobiSpectral dataset:{len(label_list)}')

        # patch_size = 64
        # xmin, ymin = 120, 300
        # xmax, ymax = xmin + patch_size, ymin + patch_size 


        cube = hdf5storage.loadmat(file, variable_names=['rad'])
        hyper = cube['rad'][:, :, bands]
        plt.imshow(hyper, cmap='gray')

def filter_rows_by_id(input_file, output_file, subject_id):
    # Convert the subject_id to a set for faster lookup
    subject_id = set(map(str, subject_id))
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header row

        # Filter rows based on the target_id
        selected_rows = [row for row in reader if row[0] in subject_id]

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header row
        writer.writerows(selected_rows)  # Write the selected rows

def make_train_test_files(labels_file, train_file, test_file, test_size=0.2):
    # Load the data from a CSV file
    data = pd.read_csv(labels_file)

    # Determine the number of rows to select for the test dataset
    test_size = int(test_size * len(data))

    # Generate indices for the test dataset by selecting every nth row
    test_indices = np.arange(0, len(data), len(data) // test_size)

    # Create test and train datasets
    test_data = data.iloc[test_indices]
    train_data = data.drop(test_indices)

    # Save the datasets to CSV files
    test_data.to_csv(test_file, index=False)
    train_data.to_csv(train_file, index=False)



def load_model(path):
    # if file not found, raise FileNotFoundError
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
            return model
    except FileNotFoundError("File not found"):
        print("File not found")

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


# if __name__ == '__main__':
#     #data = np.random.rand(3, 2)
#     data_root = opt.data_root
#     train_data = TrainDataset(data_root)
#     test_data = TestDataset(data_root)
#     X, y = [], []
#     for label, sig in train_data:
#         X.append(sig)
#         y.append(label)

#     for label, sig in test_data:
#         X.append(sig)
#         y.append(label)

#     X = np.asarray(X)
#     y = np.asarray(y)

#     correlations, p = select_feature_PCC(X, y)
#     show_correlation_P(correlations, p)

#     # show_dataset_heatmap(X)
#     show_dataset_multiplot(X)



def evaluate_image_Overlapped_CEG_SEG(test_data, model=None, path=None, scaler_X=None, scaler_y=None, show=True, save_path=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)


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
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        if scaler_X is not None:
            X_test = scaler_X.transform(X_test)
        y_pred = model.predict(X_test)
        y_pred = cp.asnumpy(y_pred)
        y_pred_mean = y_pred.mean()
        if scaler_y is not None:
            y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, test_img['label'][0])


    plot_overlapped_error_grids(predictions, labels, '../graph/SEG_background/seg600.png', show=show, save_path=save_path)

