import argparse
import logging

import numpy as np
import random
import h5py
import hdf5storage
from imageio import imread
import cv2
import torch
import torch.nn.functional as F

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

# parser = argparse.ArgumentParser(description="Spectral Regression Toolbox")
# parser.add_argument('--data_root', type=str, default='../datasets/dataset_skin/regression')
# parser.add_argument('--file', type=str, default='labels_s2_train.txt')
# opt = parser.parse_args()

class ImageList():
    def __init__(self, data_root, file, dataset, step=None, index=None, fixed_size=False):
        if fixed_size:
            if dataset == 'mobi':
                print('Loading Mobile dataset')
                self.load_mobi_fixed_size(data_root, file, step)
                self.dataset = 'mobi'
            elif dataset == 'hyper':
                print('Loading hyper dataset')
                self.load_hyper_fixed_size(data_root, file, step)
                self.dataset = 'hyper'
        else:
            if dataset == 'mobi':
                print('Loading Mobile dataset')
                self.load_mobi(data_root, file, step)
                self.dataset = 'mobi'
            elif dataset == 'hyper':
                print('Loading hyper dataset')
                self.load_hyper(data_root, file, step, index)
                self.dataset = 'hyper'

        self.data_root = data_root
        self.file = file

    # def load_mobi(self, data_root, file, step):
    #     self.image_list = []
    #     #bgr_data_path = f'{data_root}/RGBN/'
    #     # hyper_data_path = f'{data_root}/reconstructed/'
    #     hyper_data_path = f'{data_root}/rec_50/'
    #     bgr_data_path = f'{data_root}/aligned/'
    #     test_data_path = f'{data_root}/testing/'
    #     # mask_data_path = f'{data_root}/masks/'
    #     with open(f'{data_root}/{file}', 'r') as fin:
    #         # Skip the first line
    #         fin.readline()
    #         lines = [line.split(',') for line in fin]
    #         # hyper_list = [l[0] + '.mat' for l in lines]
    #         hyper_list = ['SUBID' + l[0] + '_IMG'+ l[2] + '_RGB.mat' for l in lines]
    #         #hyper_list = [l[1] + '.mat' for l in lines]
    #         # print(hyper_list)
    #         # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
    #         label_list = [np.float32(l[3].replace('\n', '')) for l in lines]

    #         # For Google Pixel and ToF
    #         xmin_list = [int(l[4]) for l in lines]
    #         ymin_list = [int(l[5]) for l in lines]
    #         xmax_list = [int(l[6]) for l in lines]
    #         ymax_list = [int(l[7]) for l in lines]

    #         # # For OnSemi
    #         # # xmin, ymin, xmax, ymax
    #         # x_min = 200
    #         # y_min = 200
    #         # x_max = 264
    #         # y_max = 264

    #         # # For RaspPi-Noir
    #         # # xmin, ymin, xmax, ymax
    #         # x_min = 100
    #         # y_min = 100
    #         # x_max = 164
    #         # y_max = 164

    #         # xmin_list = [x_min] * len(hyper_list)
    #         # ymin_list = [y_min] * len(hyper_list)
    #         # xmax_list = [x_max] * len(hyper_list)
    #         # ymax_list = [y_max] * len(hyper_list)

    #         # hyper_list = ['SUBID' + l[0] + '_IMG'+ l[2] + '_RGB.mat' for l in lines if float(l[3]) <= 5]
    #         # #hyper_list = [l[1] + '.mat' for l in lines]
    #         # # print(hyper_list)
    #         # # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
    #         # label_list = [np.float32(l[3].replace('\n', '')) for l in lines if float(l[3]) <= 5]
    #         # # print(label_list)
    #         # xmin_list = [int(l[4]) for l in lines if float(l[3]) <= 5]
    #         # ymin_list = [int(l[5]) for l in lines if float(l[3]) <= 5]
    #         # xmax_list = [int(l[6]) for l in lines if float(l[3]) <= 5]
    #         # ymax_list = [int(l[7]) for l in lines if float(l[3]) <= 5]

    #     logging.info(f'len(hyper) of Mobile dataset:{len(hyper_list)}')
    #     logging.info(f'len(label) of Mobile dataset:{len(label_list)}')


    #     # xmax, ymax = xmin + patch_size, ymin + patch_size

    #     for i in range(len(hyper_list)):
    #         image = {}
    #         sig = []
    #         label = []
    #         hyper_path = hyper_data_path + hyper_list[i]
    #         #print(hyper_path)
    #         try:
    #             cube = hdf5storage.loadmat(hyper_path, variable_names=['cube'])
    #             hyper = cube['cube'][:, :, :]
    #         except:
    #             print(f"Error loading {hyper_path}")
    #             continue

    #         xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]

    #         # # Add a rectangle on the RGB image
    #         # output_path = test_data_path + hyper_list[i].replace('.mat', '.png')
    #         # image_path = bgr_data_path + hyper_list[i].replace('.mat', '.png')
    #         # draw_rectangle(image_path, output_path, xmin, ymin, xmax, ymax)


    #         # print(hyper.shape)
    #         # mask_path = mask_data_path + mask_list[i]
    #         # mask = np.int32(imread(mask_path))
    #         # mask = mask[:, :, 1]
    #         # print(mask.shape)
    #         for m in range(xmin, xmax + 1):
    #             for n in range(ymin, ymax + 1):
    #                 # if mask[m, n] == 255:

    #                 ###### Dimension of the hypercube: y, x, bands ######
    #                 norm_sig = hyper[n, m, 0:204:step]
    #                 norm_sig = (norm_sig-norm_sig.min())/(norm_sig.max()-norm_sig.min())
    #                 sig.append(norm_sig)
    #                 label.append(label_list[i])

    #         # hyper = np.asarray(hyper)
    #         # mask_indices = mask != 255
    #         # hyper[mask_indices] = 0
    #         # image['sig'] = hyper.flatten()

    #         sig = np.asarray(sig)
    #         label = np.asarray(label)
    #         image['sig'] = sig
    #         # print('Shape of sig in image {}: {}'.format(i, image['sig'].shape))
    #         # Convert mmol/L to mg/dL
    #         image['label'] = label * 18.0182
    #         # print('Label of image {}: {}'.format(i, image['label'].shape))
    #         self.image_list.append(image)

    # def load_hyper(self, data_root, file, step=None, index=None):
    #     self.image_list = []
    #     hyper_data_path = f'{data_root}/HS_GT/'
    #     bgr_data_path = f'{data_root}/RGBN/'
    #     test_data_path = f'{data_root}/testing/'
    #     # mask_data_path = f'{data_root}/masks/'
    #     with open(f'{data_root}/{file}', 'r') as fin:
    #         # Skip the first line
    #         fin.readline()
    #         lines = [line.split(',') for line in fin]

    #         # Pick the specific subject from the list if given, otherwise use all
    #         # if subid is not None:
    #         #     # if the first element on the row is contained in the subid list
    #         #     lines = [l for l in lines if l[0] in subid]
            
    #         hyper_list = [l[1] + '.mat' for l in lines]
    #         # print(hyper_list)
    #         # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
    #         label_list = [np.float32(l[2].replace('\n', '')) for l in lines]
    #         # print(label_list)
    #         # bgr_list = [line.replace('.mat','_RGB.png') for line in hyper_list]
    #         xmin_list = [int(l[3]) for l in lines]
    #         ymin_list = [int(l[4]) for l in lines]
    #         xmax_list = [int(l[5]) for l in lines]
    #         ymax_list = [int(l[6]) for l in lines]
    #         mask_list = [line.replace('.mat', '.png') for line in hyper_list]
    #     logging.info(f'len(hyper) of Hyperspectral dataset:{len(hyper_list)}')
    #     logging.info(f'len(label) of Hyperspectral dataset:{len(label_list)}')

    #     # patch_size = 64

    #     # # Be careful to the coordinate, it is opposite to intuitive and other library
    #     # xmin, ymin = 300, 120
    #     # xmax, ymax = xmin + patch_size, ymin + patch_size 

    #     for i in range(len(hyper_list)):
    #         image = {}
    #         sig = []
    #         label = []
    #         hyper_path = hyper_data_path + hyper_list[i]
    #         cube = hdf5storage.loadmat(hyper_path, variable_names=['rad'])
    #         if index is not None:
    #             hyper = cube['rad'][:, :, index]
    #         elif step is not None:
    #             hyper = cube['rad'][:, :, 0:204:step]
    #         # hyper = np.delete(hyper, np.arange(182, 186), axis=2)

    #         xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]

    #         # Add a rectangle on the RGB image
    #         # output_path = test_data_path + hyper_list[i].replace('.mat', '.png')
    #         # image_path = bgr_data_path + hyper_list[i].replace('.mat', '_RGB.png')
    #         # draw_rectangle(image_path, output_path, xmin, ymin, xmax, ymax)

    #         # print(hyper.shape)
    #         # mask_path = mask_data_path + mask_list[i]
    #         # mask = np.int32(imread(mask_path))
    #         # mask = mask[:, :, 1]
    #         # print(mask.shape)
    #         for m in range(xmin, xmax + 1):
    #             for n in range(ymin, ymax + 1):
    #                     norm_sig = hyper[n, m, :]
    #                     # This happens when only one band is selected
    #                     if norm_sig.min() != norm_sig.max():
    #                         norm_sig = (norm_sig-norm_sig.min())/(norm_sig.max()-norm_sig.min())
    #                     sig.append(norm_sig)
    #                     label.append(label_list[i])

    #         # hyper = np.asarray(hyper)
    #         # mask_indices = mask != 255
    #         # hyper[mask_indices] = 0
    #         # image['sig'] = hyper.flatten()

    #         sig = np.asarray(sig)
    #         label = np.asarray(label)
    #         image['sig'] = sig
    #         # print('Shape of sig in image {}: {}'.format(i, image['sig'].shape))
    #         # Convert mmol/L to mg/dL
    #         image['label'] = label * 18.0182
    #         # print('Label of image {}: {}'.format(i, image['label'].shape))
    #         self.image_list.append(image)


    # A more efficient way to load the data than the above, the above is correct but too slow
    def load_mobi(self, data_root, file, step):
        self.image_list = []
        #bgr_data_path = f'{data_root}/RGBN/'
        # hyper_data_path = f'{data_root}/reconstructed/'
        hyper_data_path = f'{data_root}/rec_50/'
        bgr_data_path = f'{data_root}/aligned/'
        test_data_path = f'{data_root}/testing/'
        # mask_data_path = f'{data_root}/masks/'
        with open(f'{data_root}/{file}', 'r') as fin:
            # Skip the first line
            fin.readline()
            lines = [line.split(',') for line in fin]
            # hyper_list = [l[0] + '.mat' for l in lines]
            hyper_list = ['SUBID' + l[0] + '_IMG'+ l[2] + '_RGB.mat' for l in lines]
            #hyper_list = [l[1] + '.mat' for l in lines]
            # print(hyper_list)
            # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
            label_list = [np.float32(l[3].replace('\n', '')) for l in lines]

            # For Google Pixel and ToF
            xmin_list = [int(l[4]) for l in lines]
            ymin_list = [int(l[5]) for l in lines]
            xmax_list = [int(l[6]) for l in lines]
            ymax_list = [int(l[7]) for l in lines]

            # # For OnSemi
            # # xmin, ymin, xmax, ymax
            # x_min = 200
            # y_min = 200
            # x_max = 264
            # y_max = 264

            # # For RaspPi-Noir
            # # xmin, ymin, xmax, ymax
            # x_min = 100
            # y_min = 100
            # x_max = 164
            # y_max = 164

            # xmin_list = [x_min] * len(hyper_list)
            # ymin_list = [y_min] * len(hyper_list)
            # xmax_list = [x_max] * len(hyper_list)
            # ymax_list = [y_max] * len(hyper_list)

            # hyper_list = ['SUBID' + l[0] + '_IMG'+ l[2] + '_RGB.mat' for l in lines if float(l[3]) <= 5]
            # #hyper_list = [l[1] + '.mat' for l in lines]
            # # print(hyper_list)
            # # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
            # label_list = [np.float32(l[3].replace('\n', '')) for l in lines if float(l[3]) <= 5]
            # # print(label_list)
            # xmin_list = [int(l[4]) for l in lines if float(l[3]) <= 5]
            # ymin_list = [int(l[5]) for l in lines if float(l[3]) <= 5]
            # xmax_list = [int(l[6]) for l in lines if float(l[3]) <= 5]
            # ymax_list = [int(l[7]) for l in lines if float(l[3]) <= 5]

        logging.info(f'len(hyper) of Mobile dataset:{len(hyper_list)}')
        logging.info(f'len(label) of Mobile dataset:{len(label_list)}')

        # xmax, ymax = xmin + patch_size, ymin + patch_size

        for i in range(len(hyper_list)):
            image = {}
            hyper_path = hyper_data_path + hyper_list[i]
            #print(hyper_path)
            try:
                cube = hdf5storage.loadmat(hyper_path, variable_names=['cube'])
                hyper = cube['cube'][:, :, :]
            except:
                print(f"Error loading {hyper_path}")
                continue

            xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]

            # # Add a rectangle on the RGB image
            # output_path = test_data_path + hyper_list[i].replace('.mat', '.png')
            # image_path = bgr_data_path + hyper_list[i].replace('.mat', '.png')
            # draw_rectangle(image_path, output_path, xmin, ymin, xmax, ymax)

            # Vectorized extraction of spectral signatures from the hypercube
            region = hyper[ymin:ymax+1, xmin:xmax+1, 0:204:step]
            # Note: Dimension of the hypercube: y, x, bands
            # To mimic the loop order: for m in range(xmin, xmax+1) then for n in range(ymin, ymax+1)
            # we transpose the region to have x as the first axis and y as the second axis.
            region = np.transpose(region, (1, 0, 2))
            # Normalize each pixel's signature
            region_min = region.min(axis=2, keepdims=True)
            region_max = region.max(axis=2, keepdims=True)
            region_norm = (region - region_min) / (region_max - region_min)
            sig = region_norm.reshape(-1, region_norm.shape[2])
            label = np.full((sig.shape[0], ), label_list[i])
            image['sig'] = sig
            # Convert mmol/L to mg/dL
            image['label'] = label * 18.0182
            self.image_list.append(image)

    def load_hyper(self, data_root, file, step=None, index=None):
        self.image_list = []
        hyper_data_path = f'{data_root}/HS_GT/'
        bgr_data_path = f'{data_root}/RGBN/'
        test_data_path = f'{data_root}/testing/'
        # mask_data_path = f'{data_root}/masks/'
        with open(f'{data_root}/{file}', 'r') as fin:
            # Skip the first line
            fin.readline()
            lines = [line.split(',') for line in fin]

            # Pick the specific subject from the list if given, otherwise use all
            # if subid is not None:
            #     # if the first element on the row is contained in the subid list
            #     lines = [l for l in lines if l[0] in subid]
            
            hyper_list = [l[1] + '.mat' for l in lines]
            # print(hyper_list)
            # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
            label_list = [np.float32(l[2].replace('\n', '')) for l in lines]
            # print(label_list)
            # bgr_list = [line.replace('.mat','_RGB.png') for line in hyper_list]
            xmin_list = [int(l[3]) for l in lines]
            ymin_list = [int(l[4]) for l in lines]
            xmax_list = [int(l[5]) for l in lines]
            ymax_list = [int(l[6]) for l in lines]
            mask_list = [line.replace('.mat', '.png') for line in hyper_list]
        logging.info(f'len(hyper) of Hyperspectral dataset:{len(hyper_list)}')
        logging.info(f'len(label) of Hyperspectral dataset:{len(label_list)}')

        # patch_size = 64

        # # Be careful to the coordinate, it is opposite to intuitive and other library
        # xmin, ymin = 300, 120
        # xmax, ymax = xmin + patch_size, ymin + patch_size 

        for i in range(len(hyper_list)):
            image = {}
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path, variable_names=['rad'])
            if index is not None:
                hyper = cube['rad'][:, :, index]
                # Add a new axis to maintain consistent 3D shape for vectorized processing
                hyper = hyper[:, :, np.newaxis]
            elif step is not None:
                hyper = cube['rad'][:, :, 0:204:step]
            # hyper = np.delete(hyper, np.arange(182, 186), axis=2)

            xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]

            # Add a rectangle on the RGB image
            # output_path = test_data_path + hyper_list[i].replace('.mat', '.png')
            # image_path = bgr_data_path + hyper_list[i].replace('.mat', '_RGB.png')
            # draw_rectangle(image_path, output_path, xmin, ymin, xmax, ymax)

            # Vectorized extraction of spectral signatures from the hypercube
            region = hyper[ymin:ymax+1, xmin:xmax+1, :]
            # To mimic the loop order: for m in range(xmin, xmax+1) then for n in range(ymin, ymax+1)
            region = np.transpose(region, (1, 0, 2))
            # This happens when only one band is selected: conditionally normalize each signature
            region_min = region.min(axis=2, keepdims=True)
            region_max = region.max(axis=2, keepdims=True)
            normalized_region = np.where(region_max > region_min,
                                        (region - region_min) / (region_max - region_min),
                                        region)
            sig = normalized_region.reshape(-1, normalized_region.shape[2])
            label = np.full((sig.shape[0], ), label_list[i])
            image['sig'] = sig
            # Convert mmol/L to mg/dL
            image['label'] = label * 18.0182
            self.image_list.append(image)

    def load_mobi_fixed_size(self, data_root, file, step):
        self.image_list = []
        hyper_data_path = f'{data_root}/rec_50/'
        bgr_data_path = f'{data_root}/aligned/'
        test_data_path = f'{data_root}/testing/'
        with open(f'{data_root}/{file}', 'r') as fin:
            # Skip the first line
            fin.readline()
            lines = [line.split(',') for line in fin]
            hyper_list = ['SUBID' + l[0] + '_IMG'+ l[2] + '_RGB.mat' for l in lines]
            label_list = [np.float32(l[3].replace('\n', '')) for l in lines]
            xmin_list = [int(l[4]) for l in lines]
            ymin_list = [int(l[5]) for l in lines]
            xmax_list = [int(l[6]) for l in lines]
            ymax_list = [int(l[7]) for l in lines]

        logging.info(f'len(hyper) of Mobile dataset:{len(hyper_list)}')
        logging.info(f'len(label) of Mobile dataset:{len(label_list)}')

        for i in range(len(hyper_list)):
            image = {}
            hyper_path = hyper_data_path + hyper_list[i]
            try:
                cube = hdf5storage.loadmat(hyper_path, variable_names=['cube'])
                hyper = cube['cube'][:, :, :]
            except:
                print(f"Error loading {hyper_path}")
                continue

            xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]

            # Extract the region of interest
            region = hyper[ymin:ymax+1, xmin:xmax+1, 0:204:step]

            # Permute region from (H, W, C) to (C, H, W)
            region_permuted = np.transpose(region, (2, 0, 1))  # shape: (C, H, W)

            # Convert to tensor and create a 5D tensor with shape (1, C, 1, H, W)
            tensor_region = torch.tensor(region_permuted).unsqueeze(0).unsqueeze(2).float()  # shape: (1, C, 1, H, W)

            # Get channels count (depth) from region
            channels = tensor_region.shape[1]

            # Apply adaptive average pooling to reduce spatial dimensions to 64x64 while keeping the channel dimension unchanged.
            # Output shape should be (1, C, 1, 64, 64)
            pooled_region = F.adaptive_avg_pool3d(tensor_region, (1, 64, 64))

            # Remove the batch and dummy dimensions, resulting in shape (C, 64, 64)
            preprocessed_region = pooled_region.squeeze(0).squeeze(1).numpy()

            # Normalize the preprocessed region
            region_min = preprocessed_region.min(axis=2, keepdims=True)
            region_max = preprocessed_region.max(axis=2, keepdims=True)
            region_norm = (preprocessed_region - region_min) / (region_max - region_min)

            image['sig'] = region_norm
            image['label'] = label_list[i] * 18.0182
            self.image_list.append(image)

    def load_hyper_fixed_size(self, data_root, file, step):
        self.image_list = []
        hyper_data_path = f'{data_root}/HS_GT/'
        with open(f'{data_root}/{file}', 'r') as fin:
            # Skip the first line
            fin.readline()
            lines = [line.split(',') for line in fin]
            
            hyper_list = [l[1] + '.mat' for l in lines]
            label_list = [np.float32(l[2].replace('\n', '')) for l in lines]
            xmin_list = [int(l[4]) for l in lines]
            ymin_list = [int(l[5]) for l in lines]
            xmax_list = [int(l[6]) for l in lines]
            ymax_list = [int(l[7]) for l in lines]

        logging.info(f'len(hyper) of Hyperspectral dataset:{len(hyper_list)}')
        logging.info(f'len(label) of Hyperspectral dataset:{len(label_list)}')

        for i in range(len(hyper_list)):
            image = {}
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path, variable_names=['rad'])
            hyper = cube['rad'][:, :, :]

            xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]
            
            # Extract the region of interest
            region = hyper[ymin:ymax+1, xmin:xmax+1, 0:204:step]

            # Permute region from (H, W, C) to (C, H, W)
            region_permuted = np.transpose(region, (2, 0, 1))  # shape: (C, H, W)

            # Convert to tensor and create a 5D tensor with shape (1, C, 1, H, W)
            tensor_region = torch.tensor(region_permuted).unsqueeze(0).unsqueeze(2).float()  # shape: (1, C, 1, H, W)

            # Get channels count (depth) from region
            channels = tensor_region.shape[1]

            # Apply adaptive average pooling to reduce spatial dimensions to 64x64 while keeping the channel dimension unchanged.
            # Output shape should be (1, C, 1, 64, 64)
            pooled_region = F.adaptive_avg_pool3d(tensor_region, (1, 64, 64))

            # Remove the batch and dummy dimensions, resulting in shape (C, 64, 64)
            preprocessed_region = pooled_region.squeeze(0).squeeze(1).numpy()

            # Normalize the preprocessed region
            region_min = preprocessed_region.min(axis=2, keepdims=True)
            region_max = preprocessed_region.max(axis=2, keepdims=True)
            region_norm = (preprocessed_region - region_min) / (region_max - region_min)

            image['sig'] = region_norm
            image['label'] = label_list[i] * 18.0182
            self.image_list.append(image)


    def get_low_signal(self, band_num, threshold):
        # Create an empty set
        image_set = set()

        data_root = self.data_root
        file = self.file

        self.image_list = []
        if self.dataset == 'hyper':
            hyper_data_path = f'{data_root}/HS_GT/'
        elif self.dataset == 'mobi':
            hyper_data_path = f'{data_root}/reconstructed/'

        with open(f'{data_root}/{file}', 'r') as fin:
            # Skip the first line
            fin.readline()
            lines = [line.split(',') for line in fin]

            # Pick the specific subject from the list if given, otherwise use all
            # if subid is not None:
            #     # if the first element on the row is contained in the subid list
            #     lines = [l for l in lines if l[0] in subid]
            
            if self.dataset == 'mobi':
                hyper_list = ['SUBID' + l[0] + '_IMG'+ l[2] + '_RGB_D.mat' for l in lines]
                #hyper_list = [l[1] + '.mat' for l in lines]
                # print(hyper_list)
                # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
                label_list = [np.float32(l[3].replace('\n', '')) for l in lines]
                # print(label_list)
                xmin_list = [int(l[4]) for l in lines]
                ymin_list = [int(l[5]) for l in lines]
                xmax_list = [int(l[6]) for l in lines]
                ymax_list = [int(l[7]) for l in lines]
            elif self.dataset == 'hyper':
                hyper_list = [l[1] + '.mat' for l in lines]
                # print(hyper_list)
                # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
                label_list = [np.float32(l[2].replace('\n', '')) for l in lines]
                # print(label_list)
                # bgr_list = [line.replace('.mat','_RGB.png') for line in hyper_list]
                xmin_list = [int(l[4]) for l in lines]
                ymin_list = [int(l[5]) for l in lines]
                xmax_list = [int(l[6]) for l in lines]
                ymax_list = [int(l[7]) for l in lines]
        logging.info(f'len(hyper) of Hyperspectral dataset:{len(hyper_list)}')
        logging.info(f'len(label) of Hyperspectral dataset:{len(label_list)}')

        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if self.dataset == 'mobi':
                cube = hdf5storage.loadmat(hyper_path, variable_names=['cube'])
                hyper = cube['cube'][:, :, :]
            elif self.dataset == 'hyper':
                cube = hdf5storage.loadmat(hyper_path, variable_names=['rad'])
                hyper = cube['rad'][:, :, :]

            xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]

            for m in range(xmin, xmax + 1):
                for n in range(ymin, ymax + 1):
                    # if mask[m, n] == 255:
                        signature = hyper[n, m, band_num]
                        if signature < threshold:
                            image_set.add(i)
                            break

        return image_set
    

    def __getitem__(self, idx):
        return self.image_list[idx]

    def __len__(self):
        return len(self.image_list)

    def get_list(self):
        return self.image_list
    
def draw_rectangle(image_path, output_path, xmin, ymin, xmax, ymax):
    RGB_image = cv2.imread(image_path)
    # Define the points of the rectangle
    # Column first then row for OPENCV
    start_point = (xmin, ymin)
    end_point = (xmax, ymax)
    color = (0,0,255)
    thickness = 2
    # Draw the rectangle on the image
    RGB_image = cv2.rectangle(RGB_image, start_point, end_point, color, thickness)
    # Save the image
    cv2.imwrite(output_path, RGB_image)


        
        


# if __name__ == '__main__':
#     data_root = opt.data_root
#     file = opt.file
#     image_list = ImageList(data_root, file, 1)

