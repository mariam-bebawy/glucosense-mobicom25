import argparse
import logging

import numpy as np
import random
import h5py
import hdf5storage
from imageio import imread
import cv2


def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

# parser = argparse.ArgumentParser(description="Spectral Regression Toolbox")
# parser.add_argument('--data_root', type=str, default='../datasets/dataset_skin/regression')
# parser.add_argument('--file', type=str, default='labels_s2_train.txt')
# opt = parser.parse_args()

class ImageList():
    def __init__(self, data_root, file, step, dataset):
        if dataset == 'mobi':
            print('Loading Mobile dataset')
            self.load_mobi(data_root, file, step)
            self.dataset = 'mobi'
        elif dataset == 'hyper':
            print('Loading hyper dataset')
            self.load_hyper(data_root, file, step)
            self.dataset = 'hyper'

        self.data_root = data_root
        self.file = file

    def load_mobi(self, data_root, file, step):
        self.image_list = []
        #bgr_data_path = f'{data_root}/RGBN/'
        hyper_data_path = f'{data_root}/rec_50_new/'
        bgr_data_path = f'{data_root}/aligned/'
        test_data_path = f'{data_root}/testing/'
        # mask_data_path = f'{data_root}/masks/'
        with open(f'{data_root}/reg_split/{file}', 'r') as fin:
            # Skip the first line
            fin.readline()
            lines = [line.split(',') for line in fin]
            # hyper_list = [l[0] + '.mat' for l in lines]
            hyper_list = ['SUBID' + l[0] + '_IMG'+ l[2] + '_RGB_NoIR.mat' for l in lines]
            #hyper_list = [l[1] + '.mat' for l in lines]
            # print(hyper_list)
            # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
            label_list = [np.float32(l[3].replace('\n', '')) for l in lines]
            # print(label_list)
            xmin_list = [int(l[4]) for l in lines]
            ymin_list = [int(l[5]) for l in lines]
            xmax_list = [int(l[6]) for l in lines]
            ymax_list = [int(l[7]) for l in lines]
        logging.info(f'len(hyper) of Mobile dataset:{len(hyper_list)}')
        logging.info(f'len(label) of Mobile dataset:{len(label_list)}')

        # patch_size = 32
        # #xmin, ymin = 120, 300 #HS
        # #xmin, ymin = 60, 90  #Google Pixel
        # #xmin, ymin = 900, 400 #Onsemi
        # xmin, ymin = 70, 10 #ToF

        # xmax, ymax = xmin + patch_size, ymin + patch_size

        for i in range(len(hyper_list)):
            image = {}
            sig = []
            label = []
            hyper_path = hyper_data_path + hyper_list[i]
            #print(hyper_path)
            cube = hdf5storage.loadmat(hyper_path, variable_names=['cube'])
            hyper = cube['cube'][:, :, :]
            #print(hyper.shape)

            xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]

            #xmax = xmin+16
            xmin = 200 #onsemi
            ymin = 200  
            xmax = 264 #onsemi
            ymax = 264
            xmin = 100
            ymin = 100
            #xmax = 100+hyper.shape[1]-1
            #ymax = 100+hyper.shape[0]-1
            xmax = 164
            ymax = 164
            #ymax = ymin+16
            # Add a rectangle on the RGB image
            test_path = test_data_path + hyper_list[i].replace('.mat', '.png')
            RGB_image = cv2.imread(bgr_data_path + hyper_list[i].replace('.mat', '') + '.png')
            # image_height, image_width, _ = RGB_image.shape
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            color = (0,0,255)
            thickness = 2
            # Draw the rectangle on the image
            RGB_image = cv2.rectangle(RGB_image, start_point, end_point, color, thickness)
            # Save the image
            cv2.imwrite(test_path, RGB_image)

            # print(hyper.shape)
            # mask_path = mask_data_path + mask_list[i]
            # mask = np.int32(imread(mask_path))
            # mask = mask[:, :, 1]
            # print(mask.shape)
            for m in range(xmin, xmax + 1):
                for n in range(ymin, ymax + 1):
                    # if mask[m, n] == 255:

                    ###### Dimension of the hypercube: y, x, bands ######
                    norm_sig = hyper[n, m, :]
                    norm_sig = (norm_sig-norm_sig.min())/(norm_sig.max()-norm_sig.min())
                    
                    sig.append(norm_sig)
                    label.append(label_list[i])

            # hyper = np.asarray(hyper)
            # mask_indices = mask != 255
            # hyper[mask_indices] = 0
            # image['sig'] = hyper.flatten()

            sig = np.asarray(sig)
            label = np.asarray(label)
            image['sig'] = sig
            # print('Shape of sig in image {}: {}'.format(i, image['sig'].shape))
            # Convert mmol/L to mg/dL
            image['label'] = label * 18.0182
            # print('Label of image {}: {}'.format(i, image['label'].shape))
            self.image_list.append(image)

    def load_hyper(self, data_root, file, step):
        self.image_list = []
        #subject_id = ['1', '5', '9', '10', '11', '12', '18', '25', '26', '29']
        #subject_id = ['25']
        hyper_data_path = f'{data_root}/HS_GT/'
        bgr_data_path = f'{data_root}/RGBN/'
        test_data_path = f'{data_root}/testing/'
        # mask_data_path = f'{data_root}/masks/'
        with open(f'{data_root}/reg_split/{file}', 'r') as fin:
            # Skip the first line
            fin.readline()
            #fin = [line for line in fin if line[0] in subject_id]
            #for line in fin: 
            #    print(line)
            lines = [line.split(',') for line in fin]
            #lines = [l for l in lines if l[0] in subject_id]

            #for l in lines:
            #    print(l)
            #print(lines[1])

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
            sig = []
            label = []
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path, variable_names=['rad'])
            BANDS = [183, 184, 181, 182, 159, 22, 130, 27, 177, 34, 43, 25, 69, 42, 191, 54, 57, 37, 0, 162, 1, 40, 179, 161, 160, 96, 62, 23, 41, 156, 157, 44, 26, 45, 155, 68, 56, 94, 97, 66, 158, 80, 53, 24, 198, 36, 38, 21, 138, 2, 167, 88, 29, 67, 70, 151, 55, 90, 174, 17, 202, 188, 133, 31, 200, 140, 185, 63, 32, 145, 100, 84, 197, 193, 168, 20, 144, 154, 33, 89, 152, 60, 16, 178, 142, 15, 30, 35, 201, 46, 64, 203, 19, 71, 28, 58, 52, 72, 173, 194, 91, 139, 150, 141, 123, 164, 153, 187, 136, 128, 199, 148, 149, 129, 163, 180, 98, 95, 39, 127, 146, 189, 186, 125, 147, 165, 65, 61, 48, 74, 171, 116, 3, 73, 109, 135, 18, 75, 137, 190, 122, 93, 176, 175, 50, 192, 143, 77, 195, 134, 47, 78, 114, 170, 104, 51, 81, 115, 99, 92, 79, 107, 83, 118, 172, 111, 105, 132, 196, 76, 103, 87, 166, 131, 113, 110, 101, 119, 102, 169, 86, 121, 112, 4, 126, 124, 14, 106, 5, 59, 120, 6, 117, 85, 49, 108, 11, 82, 13, 12, 8, 10, 7, 9]
            b = np.sort(BANDS[0:12])
            hyper = cube['rad'][:, :, b]
            # remove four damaged bands [182, 183, 184, 185]
            #hyper = np.delete(hyper,np.s_[182:186], 2)
            #print(hyper.shape)

            xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]

            #xmax = xmin+16
            #ymax = ymin+16

            # Add a rectangle on the RGB image
            test_path = test_data_path + hyper_list[i].replace('.mat', '.png')
            RGB_image = cv2.imread(bgr_data_path + hyper_list[i].replace('.mat', '') + '_RGB.png')
            # Define the points of the rectangle
            # Column first then row for OPENCV
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            color = (0,0,255)
            thickness = 2
            # Draw the rectangle on the image
            RGB_image = cv2.rectangle(RGB_image, start_point, end_point, color, thickness)
            # Save the image
            cv2.imwrite(test_path, RGB_image)

            # print(hyper.shape)
            # mask_path = mask_data_path + mask_list[i]
            # mask = np.int32(imread(mask_path))
            # mask = mask[:, :, 1]
            # print(mask.shape)
            for m in range(xmin, xmax + 1):
                for n in range(ymin, ymax + 1):
                    # if mask[m, n] == 255:
                        norm_sig = hyper[n, m, 0:204:step]
                        norm_sig = (norm_sig-norm_sig.min())/(norm_sig.max()-norm_sig.min())
                        sig.append(norm_sig)
                        label.append(label_list[i])

            # hyper = np.asarray(hyper)
            # mask_indices = mask != 255
            # hyper[mask_indices] = 0
            # image['sig'] = hyper.flatten()

            sig = np.asarray(sig)
            label = np.asarray(label)
            image['sig'] = sig
            # print('Shape of sig in image {}: {}'.format(i, image['sig'].shape))
            # Convert mmol/L to mg/dL
            image['label'] = label * 18.0182
            # print('Label of image {}: {}'.format(i, image['label'].shape))
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

# if __name__ == '__main__':
#     data_root = opt.data_root
#     file = opt.file
#     image_list = ImageList(data_root, file, 1)

