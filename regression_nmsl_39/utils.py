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
from sklearn.preprocessing import RobustScaler

os.chdir("/local-scratch/GlucoseProject/mobicom23_mobispectral/regression")
from dict_dataset import ImageList
from evaluate import *
from config import *

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

# Get all training and testing data, and return the corresponding image_list
def prepare_data(data_root, file_train, file_test, dataset, step=None, index=None):
    # Build training data using list comprehensions
    image_list1 = ImageList(data_root, file_train, dataset, step, index)
    X_train = [sig for image in image_list1 for sig in image['sig']]
    y_train = [label for image in image_list1 for label in image['label']]
    
    # Build testing data using list comprehensions
    image_list2 = ImageList(data_root, file_test, dataset, step, index)
    X_test = [sig for image in image_list2 for sig in image['sig']]
    y_test = [label for image in image_list2 for label in image['label']]
    
    # Convert lists to NumPy arrays
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    logging.info('Shape of X_train: {}'.format(X_train.shape))
    logging.info('Shape of y_train: {}'.format(y_train.shape))
    logging.info('Shape of X_test: {}'.format(X_test.shape))
    logging.info('Shape of y_test: {}'.format(y_test.shape))

    return X_train, y_train, X_test, y_test, image_list2

# Get all training and testing data, and return the corresponding image_list
def prepare_data2(data_root, file_train, file_test, dataset, step=None, index=None):
    X = []
    y = []

    image_list = ImageList(data_root, file_train, file_test, dataset, step, index)
    X = [sig for image in image_list for sig in image['sig']]
    y = [label for image in image_list for label in image['label']]

    X = np.asarray(X)
    y = np.asarray(y)

    logging.info('Shape of X: {}'.format(X.shape))
    logging.info('Shape of y: {}'.format(y.shape))

    return X, y, image_list


def prepare_data_fixed_size(data_root, file_train, file_test, dataset, step=None, index=None, fixed_size=True):
    image_list1 = ImageList(data_root, file_train, dataset, step, index, fixed_size)
    image_list2 = ImageList(data_root, file_test, dataset, step, index, fixed_size)

    # Each image['sig'] is a processed image with fixed size
    X_train = [image['sig'] for image in image_list1]
    y_train = [image['label'] for image in image_list1]
    X_test = [image['sig'] for image in image_list2]
    y_test = [image['label'] for image in image_list2]

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    logging.info('Shape of X_train: {}'.format(X_train.shape))
    logging.info('Shape of y_train: {}'.format(y_train.shape))
    logging.info('Shape of X_test: {}'.format(X_test.shape))
    logging.info('Shape of y_test: {}'.format(y_test.shape))

    return X_train, y_train, X_test, y_test, image_list2


# Get all training and testing data, and return the corresponding image_list
def get_all_data(data_root, file_train, file_test, step, dataset):
    X = []
    y = []

    image_list1 = ImageList(data_root, file_train, step, dataset)
    # for i in range(len(image_list1)):
    #     image = image_list1[i]
    #     for sig in image['sig']:
    #         # plt.plot(sig)
    #         X.append(sig)
    #     for label in image['label']:
    #         y.append(label)

    image_list2 = ImageList(data_root, file_test, step, dataset)
    # for i in range(len(image_list2)):
    #     image = image_list2[i]
    #     for sig in image['sig']:
    #         X.append(sig)
    #     for label in image['label']:
    #         y.append(label)
    X = [sig for image in image_list1 for sig in image['sig']]
    y = [label for image in image_list1 for label in image['label']]
    X_temp = [sig for image in image_list2 for sig in image['sig']]
    y_temp = [label for image in image_list2 for label in image['label']]

    X.append(X_temp)
    y.append(y_temp)

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
        label = 0

        if len(X_test.shape) == 3: # For VGG model
            # Convert the dimension to 4D to fit the VGG, e.g. (50, 64, 64) -> (1, 50, 64, 64)
            X_test = np.expand_dims(X_test, axis=0)
            y_pred = model.predict(X_test)
            y_pred_mean = y_pred.numpy()
            label = test_img['label']
        else: # For traditional ML models
            if scaler_X is not None:
                X_test = scaler_X.transform(X_test)
            y_pred = model.predict(X_test)
            y_pred = cp.asnumpy(y_pred)
            y_pred_mean = y_pred.mean()
            if scaler_y is not None:
                y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
            label = test_img['label'][0]
        ard = np.append(ard, ARD(y_pred_mean, label))
    return ard

def pred_ref_glucose_PerImage(test_data, model=None, path=None, scaler_X=None, scaler_y=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    PRED_GLUCOSE, REF_GLUCOSE = [], []

    for i in range(len(test_data)):
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        label = 0

        if len(X_test.shape) == 3:  # For VGG model
            y_pred = model.predict(X_test)
            y_pred_mean = y_pred.numpy()
            label = test_img['label']
        else:  # For traditional ML models
            if scaler_X is not None:
                X_test = scaler_X.transform(X_test)
            y_pred = model.predict(X_test)
            y_pred = cp.asnumpy(y_pred)
            y_pred_mean = y_pred.mean()
            if scaler_y is not None:
                y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
            label = test_img['label'][0]
        
        PRED_GLUCOSE = np.append(PRED_GLUCOSE, y_pred_mean)
        REF_GLUCOSE = np.append(REF_GLUCOSE, label)

    return PRED_GLUCOSE, REF_GLUCOSE



def evaluate_image_CEG(test_data, model=None, path=None, scaler_X=None, scaler_y=None, show=True, save_path=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    for i in range(len(test_data)):
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        label = 0

        if len(X_test.shape) == 3:  # For VGG model
            y_pred = model.predict(X_test)
            y_pred_mean = y_pred.numpy()
            label = test_img['label']
        else:  # For traditional ML models
            if scaler_X is not None:
                X_test = scaler_X.transform(X_test)
            y_pred = model.predict(X_test)
            y_pred = cp.asnumpy(y_pred)
            y_pred_mean = y_pred.mean()
            if scaler_y is not None:
                y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
            label = test_img['label'][0]
            
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, label)

    plt2, zone = clarke_error_grid(predictions, labels, 'Clark')
    if show:
        plt2.show()
    if save_path:
        plt2.savefig(save_path)
    plt2.clf()

    return plt2, zone

def evaluate_image_SEG(test_data, model=None, path=None, scaler_X=None, scaler_y=None, show=True, save_path=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    for i in range(len(test_data)):
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        label = 0

        if len(X_test.shape) == 3:  # For VGG model
            y_pred = model.predict(X_test)
            y_pred_mean = y_pred.numpy()
            label = test_img['label']
        else:  # For traditional ML models
            if scaler_X is not None:
                X_test = scaler_X.transform(X_test)
            y_pred = model.predict(X_test)
            y_pred = cp.asnumpy(y_pred)
            y_pred_mean = y_pred.mean()
            if scaler_y is not None:
                y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
            label = test_img['label'][0]
            
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, label)

    risk_levels_percentage = calculate_seg_risks(predictions, labels, 'Surveillance')
    if show:
        plot_surveillance_error_grid(predictions, labels, '../graph/SEG_background/seg600.png', show=show, save_path=save_path)
    
    return risk_levels_percentage

def evaluate_image_Overlapped_CEG_SEG(test_data, model=None, path=None, scaler_X=None, scaler_y=None, show=True, save_path=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    for i in range(len(test_data)):
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        label = 0

        if len(X_test.shape) == 3:  # For VGG model
            y_pred = model.predict(X_test)
            y_pred_mean = y_pred.numpy()
            label = test_img['label']
        else:  # For traditional ML models
            if scaler_X is not None:
                X_test = scaler_X.transform(X_test)
            y_pred = model.predict(X_test)
            y_pred = cp.asnumpy(y_pred)
            y_pred_mean = y_pred.mean()
            if scaler_y is not None:
                y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
            label = test_img['label'][0]
            
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, label)

    plot_overlapped_CEG_SEG(predictions, labels, '../graph/SEG_background/seg600.png', show=show, save_path=save_path)

def evaluate_image_Overlapped_PEG_SEG(test_data, model=None, diabetes_type=None, path=None, scaler_X=None, scaler_y=None, show=True, save_path=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    for i in range(len(test_data)):
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        label = 0

        if len(X_test.shape) == 3:  # For VGG model
            y_pred = model.predict(X_test)
            y_pred_mean = y_pred.numpy()
            label = test_img['label']
        else:  # For traditional ML models
            if scaler_X is not None:
                X_test = scaler_X.transform(X_test)
            y_pred = model.predict(X_test)
            y_pred = cp.asnumpy(y_pred)
            y_pred_mean = y_pred.mean()
            if scaler_y is not None:
                y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
            label = test_img['label'][0]
            
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, label)

    plot_overlapped_PEG_SEG(predictions, labels, '../graph/SEG_background/seg600.png', diabetes_type, show=show, save_path=save_path)

def evaluate_risk_SEG(test_data, model=None, path=None, patient_id=None, scaler_X=None, scaler_y=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    for i in range(len(test_data)):
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        label = 0

        if len(X_test.shape) == 3:  # For VGG model
            y_pred = model.predict(X_test)
            y_pred_mean = y_pred.numpy()
            label = test_img['label']
        else:  # For traditional ML models
            if scaler_X is not None:
                X_test = scaler_X.transform(X_test)
            y_pred = model.predict(X_test)
            y_pred = cp.asnumpy(y_pred)
            y_pred_mean = y_pred.mean()
            if scaler_y is not None:
                y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
            label = test_img['label'][0]
            
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, label)

    calculate_seg_risks(predictions, labels, patient_id)

def evaluate_image_PCC(test_data, model=None, path=None, scaler_X=None, scaler_y=None, show=True, save_path=None):
    assert((model is not None) or (path is not None)), "Please specify a model or a path to model"

    if model is None:
        # Load the trained model
        model = load_model(path)

    predictions = []
    labels = []

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    for i in range(len(test_data)):
        y_pred = None
        X_test = None
        test_img = test_data[i]
        X_test = test_img['sig']
        label = 0

        if len(X_test.shape) == 3:  # For VGG model
            y_pred = model.predict(X_test)
            y_pred_mean = y_pred.numpy()
            label = test_img['label']
        else:  # For traditional ML models
            if scaler_X is not None:
                X_test = scaler_X.transform(X_test)
            y_pred = model.predict(X_test)
            y_pred = cp.asnumpy(y_pred)
            y_pred_mean = y_pred.mean()
            if scaler_y is not None:
                y_pred_mean = scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1)).ravel()
            label = test_img['label'][0]
            
        predictions = np.append(predictions, y_pred_mean)
        labels = np.append(labels, label)

    r, p = pearsonr(predictions, labels)
    print(f"Pearson correlation coefficient: {r}")
    print(f"P-value: {p}")

    return r, p

# Shows a specific band of an image
def show_image(file, bands):
        cube = hdf5storage.loadmat(file, variable_names=['rad'])
        hyper = cube['rad'][:, :, bands]
        plt.imshow(hyper, cmap='gray')
        plt.clf()

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


def make_train_test_files(labels_file, train_file, test_file, test_size=0.3):
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

'''
    Train on all subjects except one for testing, repeat for all subjects
    temp_train_file: file to store training data (Overwrite)
    temp_test_file: file to store testing data (Overwrite)
'''
def cross_validate(model, data_root, labels_file, temp_train_file, temp_test_file, step, dataset, show=True, save_path=None, logger=None):
    label_file_path = f"{data_root}/{labels_file}"
    train_file_path = f"{data_root}/{temp_train_file}"
    test_file_path = f"{data_root}/{temp_test_file}"

    # Record the MARD for each subject
    ARD, SUB = [], []

    # Record the Zone for each subject
    ZONE_CLARK, ZONE_SURVEILLANCE = [], []

    # Step 1: Get all the unique subject IDs
    subject_id = set()
    with open(label_file_path, 'r') as infile:
        reader_label = csv.reader(infile)
        header = next(reader_label)  # Skip the header
        subject_id.update([row[0] for row in reader_label])
    
    # Step 2: Iterate through each subject, override the temp_train_file and temp_test_file\
    # For each subject, train on all other subjects and test on the current subject
    for subject in subject_id:
        SUB.append(subject)
        print(f"Subject: {subject}")
        if logger:
            logger.info(f"Subject: {subject}")
        train_ids = subject_id - {subject}

        with open(label_file_path, 'r') as infile:
            reader_label = csv.reader(infile)
            header = next(reader_label)  # Skip the header
            selected_rows_train = [row for row in reader_label if row[0] in train_ids]

        with open(label_file_path, 'r') as infile:
            reader_label = csv.reader(infile)
            header = next(reader_label)  # Skip the header
            selected_rows_test = [row for row in reader_label if row[0] == subject]

        with open(train_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Write the header row
            writer.writerows(selected_rows_train)  # Write the selected rows

        with open(test_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Write the header row
            writer.writerows(selected_rows_test)  # Write the selected rows

    # Step 3: Prepare the data and train the model
        X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, temp_train_file, temp_test_file, dataset, step)
        x_scaler = RobustScaler()
        y_scaler = RobustScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

        model.fit(X_train, y_train)
        y_pred = cp.asnumpy(model.predict(X_test))
        ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
        ard_mean = ard.mean()
        ARD.append(ard_mean)
        # print("ARD by images: ", ard)
        print("Average ARD: ", ard_mean)

        plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False, save_path=f"../graph/MLP/CEG/clark_test{subject}.png")
        print("Clark Error Grid Zone: ", zone)
        risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False, save_path=f"../graph/MLP/SEG/seg_test{subject}.png")
        print("Surveillance Error Grid Zone: ", risk_levels_percentage)

        # Save the pred, ref glucose values for each subject for plotting in CEG/SEG
        # Save it in a csv file, 3 columns: subject, pred, ref
        PRED_GLUCOSE, REF_GLUCOSE = pred_ref_glucose_PerImage(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler)
        SUBJECT = [subject] * len(PRED_GLUCOSE)
        df = pd.DataFrame({'subject': SUBJECT, 'pred': PRED_GLUCOSE, 'ref': REF_GLUCOSE})
        df.to_csv(f"./Experiment/LeaveOneOut/LeaveOneOut_ToF{subject}.csv", index=False)

        ZONE_CLARK.append(zone)
        ZONE_SURVEILLANCE.append(risk_levels_percentage)

        # evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False, save_path=f"../graph/MLP/CEG/clark_test{subject}.png")
        # evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False, save_path=f"../graph/MLP/SEG/seg_test{subject}.png")
        # CDF(ard, show=False, save_path=f"../graph/MLP/CDF/cdf_test{subject}.png")

    # Step 4: Print the MARD results across all subjects
    plt.plot(SUB, ARD)
    plt.xlabel('Subject')
    plt.ylabel('MARD')
    plt.title('MARD test on specific Subject')
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.clf()
    
    print(SUB, ARD)
    if logger:
        logger.info('done')
    print('done')

    return SUB, ARD, ZONE_CLARK, ZONE_SURVEILLANCE


'''
    Train on all subjects except one for testing, repeat for all subjects.
    For the testing subject, use half of the samples as calibration to the training set.
    temp_train_file: file to store training data (Overwrite)
    temp_test_file: file to store testing data (Overwrite)
'''
def cross_validate_calibration(model, data_root, labels_file, temp_train_file, temp_test_file, step, dataset, show=True, save_path=None, logger=None):
    label_file_path = f"{data_root}/{labels_file}"
    train_file_path = f"{data_root}/{temp_train_file}"
    test_file_path = f"{data_root}/{temp_test_file}"

    # Record the MARD for each subject
    ARD, SUB = [], []

    # Record the Zone for each subject
    ZONE_CLARK, ZONE_SURVEILLANCE = [], []

    # Step 1: Get all the unique subject IDs
    subject_id = set()
    with open(label_file_path, 'r') as infile:
        reader_label = csv.reader(infile)
        header = next(reader_label)  # Skip the header
        subject_id.update([row[0] for row in reader_label])
    
    # Step 2: Iterate through each subject, override the temp_train_file and temp_test_file
    # For each subject, train on all other subjects and half of the current subject
    for subject in subject_id:
        SUB.append(subject)
        print(f"Subject: {subject}")
        if logger:
            logger.info(f"Subject: {subject}")
        train_ids = subject_id - {subject}

        # Read all rows once to avoid reopening the file multiple times
        with open(label_file_path, 'r') as infile:
            reader_label = list(csv.reader(infile))
            header = reader_label[0]  # Header row
            all_rows = reader_label[1:]  # Data rows

        # Separate training and testing rows
        selected_rows_train = [row for row in all_rows if row[0] in train_ids]
        selected_rows_test_all = [row for row in all_rows if row[0] == subject]

        # # Split the testing subject's data into calibration (train) and testing
        # num_test = len(selected_rows_test_all)
        # half = num_test // 2
        # calibration_rows = selected_rows_test_all[:half]
        # test_rows = selected_rows_test_all[half:]

        # # Instead of picking the first half, pick every 2 rows
        # calibration_rows = selected_rows_test_all[::2]
        # test_rows = selected_rows_test_all[1::2]

        # Pick every 3 rows for testing, the remaining for calibration
        test_rows = selected_rows_test_all[::3]  # Takes indices 0,3,6,9,...
        calibration_rows = [row for i, row in enumerate(selected_rows_test_all) if i % 3 != 0]  # Takes all other rows

        # Add half of the testing subject to training
        selected_rows_train.extend(calibration_rows)

        # Write training data
        with open(train_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Write the header row
            writer.writerows(selected_rows_train)  # Write the selected rows including calibration

        # Write testing data
        with open(test_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Write the header row
            writer.writerows(test_rows)  # Write the remaining half for testing

        # Step 3: Prepare the data and train the model
        X_train, y_train, X_test, y_test, test_data = prepare_data(data_root, temp_train_file, temp_test_file, dataset, step)
        x_scaler = RobustScaler()
        y_scaler = RobustScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

        model.fit(X_train, y_train)
        y_pred = cp.asnumpy(model.predict(X_test))
        ard = ard_image(test_data, model, scaler_X=x_scaler, scaler_y=y_scaler)
        ard_mean = ard.mean()
        ARD.append(ard_mean)
        # print("ARD by images: ", ard)
        print("Average ARD: ", ard_mean)

        plt2, zone = evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False, save_path=f"../graph/MLP/CEG/clark_test{subject}.png")
        print("Clark Error Grid Zone: ", zone)
        risk_levels_percentage = evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False, save_path=f"../graph/MLP/SEG/seg_test{subject}.png")
        print("Surveillance Error Grid Zone: ", risk_levels_percentage)

        # Save the pred, ref glucose values for each subject for plotting in CEG/SEG
        # Save it in a csv file, 3 columns: subject, pred, ref
        PRED_GLUCOSE, REF_GLUCOSE = pred_ref_glucose_PerImage(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler)
        SUBJECT = [subject] * len(PRED_GLUCOSE)
        df = pd.DataFrame({'subject': SUBJECT, 'pred': PRED_GLUCOSE, 'ref': REF_GLUCOSE})
        df.to_csv(f"./Experiment/LeaveOneOut/LeaveOneOut_Calibration_3_GooglePixel{subject}.csv", index=False)

        ZONE_CLARK.append(zone)
        ZONE_SURVEILLANCE.append(risk_levels_percentage)

        # evaluate_image_CEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False, save_path=f"../graph/MLP/CEG/clark_test{subject}.png")
        # evaluate_image_SEG(test_data, model=model, scaler_X=x_scaler, scaler_y=y_scaler, show=False, save_path=f"../graph/MLP/SEG/seg_test{subject}.png")
        # CDF(ard, show=False, save_path=f"../graph/MLP/CDF/cdf_test{subject}.png")

    # Step 4: Print the MARD results across all subjects
    plt.plot(SUB, ARD)
    plt.xlabel('Subject')
    plt.ylabel('MARD')
    plt.title('MARD test on specific Subject')
    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.clf()
    
    print(SUB, ARD)
    if logger:
        logger.info('done')
    print('done')

    return SUB, ARD, ZONE_CLARK, ZONE_SURVEILLANCE
        

    

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


def clarke_error_zone_detailed(act, pred):
    """
    This function outputs the Clarke Error Grid region (encoded as integer)
    for a combination of actual and predicted value

    Based on 'Evaluating clinical accuracy of systems for self-monitoring of blood glucose':
    https://care.diabetesjournals.org/content/10/5/622
    """
    # Zone A
    if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
        return 0
    # Zone E - left upper
    if act <= 70 and pred >= 180:
        return 8
    # Zone E - right lower
    if act >= 180 and pred <= 70:
        return 7
    # Zone D - right
    if act >= 240 and 70 <= pred <= 180:
        return 6
    # Zone D - left
    if act <= 70 <= pred <= 180:
        return 5
    # Zone C - upper
    if 70 <= act <= 290 and pred >= act + 110:
        return 4
    # Zone C - lower
    if 130 <= act <= 180 and pred <= (7/5) * act - 182:
        return 3
    # Zone B - upper
    if act < pred:
        return 2
    # Zone B - lower
    return 1

def parkes_error_zone_detailed(act, pred, diabetes_type):
    """
    This function outputs the Parkes Error Grid region (encoded as integer)
    for a combination of actual and predicted value
    for type 1 and type 2 diabetic patients

    Based on the article 'Technical Aspects of the Parkes Error Grid':
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3876371/
    """
    def above_line(x_1, y_1, x_2, y_2, strict=False):
        if x_1 == x_2:
            return False

        y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
        return pred > y_line if strict else pred >= y_line

    def below_line(x_1, y_1, x_2, y_2, strict=False):
        return not above_line(x_1, y_1, x_2, y_2, not strict)

    def parkes_type_1(act, pred):
        # Zone E
        if above_line(0, 150, 35, 155) and above_line(35, 155, 50, 550):
            return 7
        # Zone D - left upper
        if (pred > 100 and above_line(25, 100, 50, 125) and
                above_line(50, 125, 80, 215) and above_line(80, 215, 125, 550)):
            return 6
        # Zone D - right lower
        if (act > 250 and below_line(250, 40, 550, 150)):
            return 5
        # Zone C - left upper
        if (pred > 60 and above_line(30, 60, 50, 80) and
                above_line(50, 80, 70, 110) and above_line(70, 110, 260, 550)):
            return 4
        # Zone C - right lower
        if (act > 120 and below_line(120, 30, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        # Zone B - left upper
        if (pred > 50 and above_line(30, 50, 140, 170) and
                above_line(140, 170, 280, 380) and (act < 280 or above_line(280, 380, 430, 550))):
            return 2
        # Zone B - right lower
        if (act > 50 and below_line(50, 30, 170, 145) and
                below_line(170, 145, 385, 300) and (act < 385 or below_line(385, 300, 550, 450))):
            return 1
        # Zone A
        return 0

    def parkes_type_2(act, pred):
        # Zone E
        if (pred > 200 and above_line(35, 200, 50, 550)):
            return 7
        # Zone D - left upper
        if (pred > 80 and above_line(25, 80, 35, 90) and above_line(35, 90, 125, 550)):
            return 6
        # Zone D - right lower
        if (act > 250 and below_line(250, 40, 410, 110) and below_line(410, 110, 550, 160)):
            return 5
        # Zone C - left upper
        if (pred > 60 and above_line(30, 60, 280, 550)):
            return 4
        # Zone C - right lower
        if (below_line(90, 0, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        # Zone B - left upper
        if (pred > 50 and above_line(30, 50, 230, 330) and
                (act < 230 or above_line(230, 330, 440, 550))):
            return 2
        # Zone B - right lower
        if (act > 50 and below_line(50, 30, 90, 80) and below_line(90, 80, 330, 230) and
                (act < 330 or below_line(330, 230, 550, 450))):
            return 1
        # Zone A
        return 0

    if diabetes_type == 1:
        return parkes_type_1(act, pred)

    if diabetes_type == 2:
        return parkes_type_2(act, pred)

    raise Exception('Unsupported diabetes type')

# clarke_error_zone_detailed = np.vectorize(clarke_error_zone_detailed)
parkes_error_zone_detailed = np.vectorize(parkes_error_zone_detailed)

def zone_accuracy(act_arr, pred_arr, mode='clarke', detailed=False, diabetes_type=1):
    """
    Calculates the average percentage of each zone based on Clarke or Parkes
    Error Grid analysis for an array of predictions and an array of actual values
    """
    acc = np.zeros(9)
    if mode == 'clarke':
        res = clarke_error_zone_detailed(act_arr, pred_arr)
    elif mode == 'parkes':
        res = parkes_error_zone_detailed(act_arr, pred_arr, diabetes_type)
    else:
        raise Exception('Unsupported error grid mode')

    acc_bin = np.bincount(res)
    acc[:len(acc_bin)] = acc_bin

    if not detailed:
        acc[1] = acc[1] + acc[2]
        acc[2] = acc[3] + acc[4]
        acc[3] = acc[5] + acc[6]
        acc[4] = acc[7] + acc[8]
        acc = acc[:5]

    return acc / sum(acc)

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



