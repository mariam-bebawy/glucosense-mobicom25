import torch
import numpy as np
import argparse
import os
import torch.backends.cudnn as cudnn
from architecture import *
from utils import AverageMeter, save_matv73, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_SAM
from hsi_dataset import TrainDataset, ValidDataset
from torch.utils.data import DataLoader
from spectral_metrics import test_msam, test_sid, test_ssim, test_psnr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--data_root', type=str, default='../../../datasets/HSDatasets/')
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default='mst_AWB_940_t50_3bands.pth')
parser.add_argument('--outf', type=str, default='../../../datasets/HSDatasets/rec_50_3/')
parser.add_argument("--gpu_id", type=str, default='0')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

# load dataset
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
if torch.cuda.is_available():
    criterion_mrae.cuda()
    criterion_rmse.cuda()

# Validate
with open(f'{opt.data_root}/split_txt/valid_test.txt', 'r') as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()
#print("HS list",hyper_list)
var_name = 'cube'
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    losses_sam = AverageMeter()
    losses_sid = AverageMeter()
    losses_ssim = AverageMeter()
    losses_psnr_new = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            if method=='awan':   # To avoid out of memory, we crop the center region as input for AWAN.
                output = model(input[:, :, 118:-118, 118:-118])
                loss_mrae = criterion_mrae(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_rmse = criterion_rmse(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
                loss_psnr = criterion_psnr(output[:, :, 10:-10, 10:-10], target[:, :, 128:-128, 128:-128])
            else:
                #print(input.shape)
                output = model(input)
                loss_mrae = criterion_mrae(output,target)
                loss_rmse = criterion_rmse(output,target)
                loss_psnr = criterion_psnr(output,target)
                #loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                #loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
                #loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)

        result = output[:, :, :].cpu().numpy() * 1.0
        result = np.transpose(np.squeeze(result), [1, 2, 0])
        result = np.minimum(result, 1.0)
        result = np.maximum(result, 0)
        mat_name = hyper_list[i]
        mat_dir = os.path.join(opt.outf, mat_name)
        save_matv73(mat_dir, var_name, result)

        gt = target[:, :, :].cpu().numpy() * 1.0
        gt = np.transpose(np.squeeze(gt), [1, 2, 0])
        gt = np.minimum(gt, 1.0)
        gt = np.maximum(gt, 0)
        loss_sam = test_msam(result,gt)
        loss_sid = test_sid(result,gt)
        loss_ssim = test_ssim(result,gt)
        loss_psnr_new = test_psnr(result,gt)
        losses_sam.update(loss_sam)
        losses_sid.update(loss_sid)
        losses_ssim.update(loss_ssim)
        losses_psnr_new.update(loss_psnr_new)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg, losses_sam.avg, losses_sid.avg, losses_ssim.avg, losses_psnr_new.avg

if __name__ == '__main__':
    cudnn.benchmark = True
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    model = model_generator(method, pretrained_model_path).cuda()
    mrae, rmse, psnr, sam, sid, ssim, psnr_new  = validate(val_loader, model)
    print(f'method:{method}, mrae:{mrae}, rmse:{rmse}, sam: {sam}, sid: {sid}, ssim: {ssim}, psnr:{psnr}, psnr_new:{psnr_new}')
