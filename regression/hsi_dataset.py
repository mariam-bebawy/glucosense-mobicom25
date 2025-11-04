from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import hdf5storage
from imageio import imread

class TrainDataset(Dataset):
    def __init__(self, data_root):
        self.sig = []
        self.label = []
        hyper_data_path = f'{data_root}/HS_GT/'
        bgr_data_path = f'{data_root}/RGBN/'
        mask_data_path = f'{data_root}/masks/'
        with open(f'{data_root}/labels_s1_train.txt', 'r') as fin:
            lines = [line.split(',') for line in fin]
            hyper_list = [l[0]+'.mat' for l in lines]
            #print(hyper_list)
            label_list = [np.float32(l[1].replace('\n','')) for l in lines]
            #print(label_list)
            #bgr_list = [line.replace('.mat','_RGB.png') for line in hyper_list]
            mask_list = [line.replace('.mat','.png') for line in hyper_list]
        print(f'len(hyper) of MobiSpectral dataset:{len(hyper_list)}')
        print(f'len(label) of MobiSpectral dataset:{len(label_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path,variable_names=['rad'])
            hyper = cube['rad'][:,:,:]
            #print(hyper.shape)
            mask_path = mask_data_path + mask_list[i]
            mask = np.int32(imread(mask_path))
            mask = mask[:,:,1]
            #print(mask.shape)
            for m in range(512):
                for n in range(512):
                    if mask[m, n] == 255:
                        self.sig.append(hyper[m, n, 1:204:3])
                        self.label.append(label_list[i])
            
            #print(len(self.sig))
            #print(len(self.label))
            #hyper = np.transpose(hyper, [2, 0, 1])
            #bgr_path = bgr_data_path + bgr_list[i]
            #nir_path = bgr_data_path + nir_list[i]
            #nir_path1 = bgr_data_path + nir_list1[i]
            #bgr = imread(bgr_path)
            #nir = imread(nir_path)
            #nir1 = imread(nir_path1)
            #bgr = np.float32(bgr)
            #nir = np.float32(nir)
            #nir1 = np.float32(nir1)
            #bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            #nir = (nir - nir.min()) / (nir.max() - nir.min())
            #nir1 = (nir1 - nir1.min()) / (nir1.max() - nir1.min())
            #bgr = np.dstack((bgri,nir,nir1))
            #bgr = np.transpose(bgr, [2, 0, 1])
            #self.sig.append(hyper)
            #self.bgrs.append(bgr)
            #print(f'MobiSpectral scene {i} is loaded.')

    def __getitem__(self, idx):
        sig = self.sig[idx]
        label = self.label[idx]
        return np.ascontiguousarray(label), np.ascontiguousarray(sig)

    def __len__(self):
        return len(self.sig)


class TestDataset(Dataset):
    def __init__(self, data_root):
        self.sig = []
        self.label = []
        hyper_data_path = f'{data_root}/HS_GT/'
        bgr_data_path = f'{data_root}/RGBN/'
        mask_data_path = f'{data_root}/masks/'
        with open(f'{data_root}/labels_s1_test.txt', 'r') as fin:
            lines = [line.split(',') for line in fin]
            hyper_list = [l[0]+'.mat' for l in lines]
            #print(hyper_list)
            label_list = [np.float32(l[1].replace('\n','')) for l in lines]
            #print(label_list)
            #bgr_list = [line.replace('.mat','_RGB.png') for line in hyper_list]
            mask_list = [line.replace('.mat','.png') for line in hyper_list]
        print(f'len(hyper) of MobiSpectral dataset:{len(hyper_list)}')
        print(f'len(label) of MobiSpectral dataset:{len(label_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path,variable_names=['rad'])
            hyper = cube['rad'][:,:,:]
            #print(hyper.shape)
            mask_path = mask_data_path + mask_list[i]
            mask = np.int32(imread(mask_path))
            mask = mask[:,:,1]
            #print(mask.shape)
            for m in range(512):
                for n in range(512):
                    if mask[m, n] == 255:
                        self.sig.append(hyper[m, n, 1:204:3])
                        self.label.append(label_list[i])

            #print(len(self.sig))
            #print(len(self.label))
            #hyper = np.transpose(hyper, [2, 0, 1])
            #bgr_path = bgr_data_path + bgr_list[i]
            #nir_path = bgr_data_path + nir_list[i]
            #nir_path1 = bgr_data_path + nir_list1[i]
            #bgr = imread(bgr_path)
            #nir = imread(nir_path)
            #nir1 = imread(nir_path1)
            #bgr = np.float32(bgr)
            #nir = np.float32(nir)
            #nir1 = np.float32(nir1)
            #bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            #nir = (nir - nir.min()) / (nir.max() - nir.min())
            #nir1 = (nir1 - nir1.min()) / (nir1.max() - nir1.min())
            #bgr = np.dstack((bgri,nir,nir1))
            #bgr = np.transpose(bgr, [2, 0, 1])
            #self.sig.append(hyper)
            #self.bgrs.append(bgr)
            #print(f'MobiSpectral scene {i} is loaded.')

    def __getitem__(self, idx):
        sig = self.sig[idx]
        label = self.label[idx]
        return np.ascontiguousarray(label), np.ascontiguousarray(sig)

    def __len__(self):
        return len(self.sig)

