from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
import hdf5storage
from imageio import imread

class TrainDataset(Dataset):
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):
        self.crop_size = crop_size
        self.hypers = []
        self.bgrs = []
        self.arg = arg
        h,w = 512,512  # img shape
        self.stride = stride
        self.patch_per_line = (w-crop_size)//stride+1
        self.patch_per_colum = (h-crop_size)//stride+1
        self.patch_per_img = self.patch_per_line*self.patch_per_colum

        hyper_data_path = f'{data_root}/HS_GT/'
        bgr_data_path = f'{data_root}/RGBN/'
        BANDS = [183, 184, 181, 182, 159, 22, 130, 27, 177, 34, 43, 25, 69, 42, 191, 54, 57, 37, 0, 162, 1, 40, 179, 161, 160, 96, 62, 23, 41, 156, 157, 44, 26, 45, 155, 68, 56, 94, 97, 66, 158, 80, 53, 24, 198, 36, 38, 21, 138, 2, 167, 88, 29, 67, 70, 151, 55, 90, 174, 17, 202, 188, 133, 31, 200, 140, 185, 63, 32, 145, 100, 84, 197, 193, 168, 20, 144, 154, 33, 89, 152, 60, 16, 178, 142, 15, 30, 35, 201, 46, 64, 203, 19, 71, 28, 58, 52, 72, 173, 194, 91, 139, 150, 141, 123, 164, 153, 187, 136, 128, 199, 148, 149, 129, 163, 180, 98, 95, 39, 127, 146, 189, 186, 125, 147, 165, 65, 61, 48, 74, 171, 116, 3, 73, 109, 135, 18, 75, 137, 190, 122, 93, 176, 175, 50, 192, 143, 77, 195, 134, 47, 78, 114, 170, 104, 51, 81, 115, 99, 92, 79, 107, 83, 118, 172, 111, 105, 132, 196, 76, 103, 87, 166, 131, 113, 110, 101, 119, 102, 169, 86, 121, 112, 4, 126, 124, 14, 106, 5, 59, 120, 6, 117, 85, 49, 108, 11, 82, 13, 12, 8, 10, 7, 9]

        with open(f'{data_root}/split_txt/train.txt', 'r') as fin:
            hyper_list = [line.replace('\n','.mat') for line in fin]
            bgr_list = [line.replace('.mat','_RGB_NoIR_AWB.png') for line in hyper_list]
            nir_list1 = [line.replace('.mat','_NIR940.png') for line in hyper_list]
            #nir_list2 = [line.replace('.mat','_NIR940.jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        nir_list1.sort()
        #nir_list2.sort()
        print(f'len(hyper) of MobiSpectral dataset:{len(hyper_list)}')
        print(f'len(bgr) of MobiSpectral dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path,variable_names=['rad'])
            b = np.sort(BANDS[0:50])
            hyper = cube['rad'][:,:,b]
            hyper = np.transpose(hyper, [2, 0, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            nir_path1 = bgr_data_path + nir_list1[i]
            #nir_path2 = bgr_data_path + nir_list2[i]
            bgr = imread(bgr_path)
            nir1 = imread(nir_path1)
            #nir2 = imread(nir_path2)
            bgr = np.float32(bgr)
            nir1 = np.float32(nir1)
            #nir2 = np.float32(nir2)
            bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
            nir1 = (nir1-nir1.min())/(nir1.max()-nir1.min())
            #nir2 = (nir2-nir2.min())/(nir2.max()-nir2.min())
            #bgr = np.dstack((bgr,nir1))
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            print(f'MobiSpectral scene {i} is loaded.')
        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img*self.img_num

class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []
        hyper_data_path = f'{data_root}/HS_GT/'
        bgr_data_path = f'{data_root}/RGBN/'
        BANDS = [183, 184, 181, 182, 159, 22, 130, 27, 177, 34, 43, 25, 69, 42, 191, 54, 57, 37, 0, 162, 1, 40, 179, 161, 160, 96, 62, 23, 41, 156, 157, 44, 26, 45, 155, 68, 56, 94, 97, 66, 158, 80, 53, 24, 198, 36, 38, 21, 138, 2, 167, 88, 29, 67, 70, 151, 55, 90, 174, 17, 202, 188, 133, 31, 200, 140, 185, 63, 32, 145, 100, 84, 197, 193, 168, 20, 144, 154, 33, 89, 152, 60, 16, 178, 142, 15, 30, 35, 201, 46, 64, 203, 19, 71, 28, 58, 52, 72, 173, 194, 91, 139, 150, 141, 123, 164, 153, 187, 136, 128, 199, 148, 149, 129, 163, 180, 98, 95, 39, 127, 146, 189, 186, 125, 147, 165, 65, 61, 48, 74, 171, 116, 3, 73, 109, 135, 18, 75, 137, 190, 122, 93, 176, 175, 50, 192, 143, 77, 195, 134, 47, 78, 114, 170, 104, 51, 81, 115, 99, 92, 79, 107, 83, 118, 172, 111, 105, 132, 196, 76, 103, 87, 166, 131, 113, 110, 101, 119, 102, 169, 86, 121, 112, 4, 126, 124, 14, 106, 5, 59, 120, 6, 117, 85, 49, 108, 11, 82, 13, 12, 8, 10, 7, 9]
        with open(f'{data_root}/split_txt/valid.txt', 'r') as fin:
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('.mat','_RGB_NoIR_AWB.png') for line in hyper_list]
            nir_list1 = [line.replace('.mat','_NIR940.png') for line in hyper_list]
            #nir_list2 = [line.replace('.mat','_NIR940.jpg') for line in hyper_list]
        hyper_list.sort()
        bgr_list.sort()
        nir_list1.sort()
        #nir_list2.sort()
        print(f'len(hyper_valid) of MobiSpectral dataset:{len(hyper_list)}')
        print(f'len(bgr_valid) of MobiSpectral dataset:{len(bgr_list)}')
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            cube = hdf5storage.loadmat(hyper_path,variable_names=['rad'])
            b = np.sort(BANDS[0:50])
            hyper = cube['rad'][:,:,b]
            hyper = np.transpose(hyper, [2, 0, 1])
            bgr_path = bgr_data_path + bgr_list[i]
            nir_path1 = bgr_data_path + nir_list1[i]
            #nir_path2 = bgr_data_path + nir_list2[i]
            bgr = imread(bgr_path)
            nir1 = imread(nir_path1)
            #nir2 = imread(nir_path2)
            bgr = np.float32(bgr)
            nir1 = np.float32(nir1)
            #nir2 = np.float32(nir2)
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())
            nir1 = (nir1 - nir1.min()) / (nir1.max() - nir1.min())
            #nir2 = (nir2 - nir2.min()) / (nir2.max() - nir2.min())

            #bgr = np.dstack((bgr,nir1))
            bgr = np.transpose(bgr, [2, 0, 1])
            self.hypers.append(hyper)
            self.bgrs.append(bgr)
            print(f'MobiSpectral scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)
