import torch
import argparse
import torch.backends.cudnn as cudnn
import os
from architecture import *
from utils import save_matv73
import glob
import cv2
import numpy as np
import itertools
import imageio

parser = argparse.ArgumentParser(description="SSR")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='./Models/mst_AWB_940_t50_3bands.pth')
parser.add_argument('--data_root', type=str, default='../../../datasets/MobileDatasets/RaspberryPi-RGB-NoIR/processed/')
parser.add_argument('--csv_root', type=str, default='../../../datasets/MobileDatasets/RaspberryPi-RGB-NoIR/')
parser.add_argument('--outf', type=str, default='../../../datasets/MobileDatasets/RaspberryPi-RGB-NoIR/rec_50_new/')
parser.add_argument('--ensemble_mode', type=str, default='mean')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def main():
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()
    test_path = os.path.join(opt.data_root)
    csv_root = opt.csv_root
    #print(test_path)
    test(model, test_path, opt.outf, csv_root)

def test(model, test_path, save_path, csv_root):
    #img_path_name = glob.glob(os.path.join(test_path, '*_RGB.png'))
    #img_path_name.sort()
    var_name = 'cube'
    with open(f'{csv_root}/labels_RaspberryPi-RGB_NoIR(all).csv', 'r') as fin:
            # Skip the first line
            fin.readline()
            lines = [line.split(',') for line in fin]
            # hyper_list = [l[0] + '.mat' for l in lines]
            image_list = ['SUBID' + l[0] + '_IMG'+ l[2] + '_RGB_NoIR.png' for l in lines]
            #hyper_list = [l[1] + '.mat' for l in lines]
            # print(hyper_list)
            # label_list = [np.float32(l[1].replace('\n', '')) for l in lines]
            #label_list = [np.float32(l[3].replace('\n', '')) for l in lines]
            # print(label_list)
            xmin_list = [int(l[4]) for l in lines]
            ymin_list = [int(l[5]) for l in lines]
            xmax_list = [int(l[6]) for l in lines]
            ymax_list = [int(l[7]) for l in lines]
    
    print(len(image_list))
    for i in range(len(image_list)):
        img_path_name = os.path.join(test_path, image_list[i])
        print(img_path_name)
        rgb = imageio.imread(img_path_name)
        print(rgb.shape)
        #nir_path = img_path_name.replace('_RGB.png','_NIR.png')
        #nir = imageio.imread(nir_path)
        rgb = np.float32(rgb)
        #rgb = rgb[:,:,:]
        #nir = np.float32(nir)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        #nir = (nir - nir.min()) / (nir.max() - nir.min())
        #nir = nir[:,:,0]
        #rgb = np.dstack((rgb, nir))
        xmin, ymin, xmax, ymax = xmin_list[i], ymin_list[i], xmax_list[i], ymax_list[i]
        rgb = rgb[ymin-100:ymax+100,xmin-100:xmax+100,:]
        #rgb = rgb[:,:,:]
        rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
        rgb = torch.from_numpy(rgb).float().cuda()
        with torch.no_grad():
            result = forward_ensemble(rgb, model, opt.ensemble_mode)
        result = result.cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        mat_name = img_path_name.split('/')[-1][:-4] + '.mat'
        mat_dir = os.path.join(save_path, mat_name)
        save_matv73(mat_dir, var_name, result)

def forward_ensemble(x, forward_func, ensemble_mode = 'mean'):
    def _transform(data, xflip, yflip, transpose, reverse=False):
        if not reverse:  # forward transform
            if xflip:
                data = torch.flip(data, [3])
            if yflip:
                data = torch.flip(data, [2])
            if transpose:
                data = torch.transpose(data, 2, 3)
        else:  # reverse transform
            if transpose:
                data = torch.transpose(data, 2, 3)
            if yflip:
                data = torch.flip(data, [2])
            if xflip:
                data = torch.flip(data, [3])
        return data

    outputs = []
    opts = itertools.product((False, True), (False, True), (False, True))
    for xflip, yflip, transpose in opts:
        data = x.clone()
        data = _transform(data, xflip, yflip, transpose)
        data = forward_func(data)
        outputs.append(
            _transform(data, xflip, yflip, transpose, reverse=True))
    if ensemble_mode == 'mean':
        return torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        return torch.stack(outputs, 0).median(0)[0]


if __name__ == '__main__':
    main()
