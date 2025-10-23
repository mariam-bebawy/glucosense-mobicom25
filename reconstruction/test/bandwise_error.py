import numpy as np
import hdf5storage
from utils import AverageMeter, save_matv73
from spectral_metrics import test_msam, test_sid, test_ssim, test_mrae, test_rmse, test_psnr, spectral_angle, spectral_divergence

data_root = '../../datasets/dataset_skin/reconstruction/'
hyper_gt_path = f'{data_root}/Train_Spec/'
hyper_inf_path = 'exp/hs_inference_skinrgbnir/'

with open(f'{data_root}/split_txt/test_list.txt', 'r') as fin:
    hyper_list = [line.replace('\n', '.mat') for line in fin]
hyper_list.sort()



avg_mrae = AverageMeter()
avg_rmse = AverageMeter()
avg_psnr = AverageMeter()
avg_sam = AverageMeter()
avg_sid = AverageMeter()
avg_ssim = AverageMeter()

avg_mrae = []
avg_rmse = []
avg_psnr = []
avg_sam = []
avg_sid = []
avg_ssim = []


for i in range(len(hyper_list)):
    gt_path = hyper_gt_path + hyper_list[i]
    rad = hdf5storage.loadmat(gt_path,variable_names=['rad'])
    hyper_gt = rad['rad'][:,:,1:204:3]

    inf_path = hyper_inf_path + hyper_list[i]
    cube = hdf5storage.loadmat(inf_path,variable_names=['cube'])
    hyper_inf = cube['cube']

    sam, sid, ssim, psnr, mrae, rmse = [], [], [], [], [], []
    for band in range(68):
        sam.append(spectral_angle(hyper_inf[:,:,band].reshape(-1),hyper_gt[:,:,band].reshape(-1)))
        ssim.append(test_ssim(hyper_inf[:,:,band],hyper_gt[:,:,band]))
        mrae.append(test_mrae(hyper_inf[:,:,band],hyper_gt[:,:,band]))
        rmse.append(test_rmse(hyper_inf[:,:,band],hyper_gt[:,:,band]))
        psnr.append(test_psnr(hyper_inf[:,:,band],hyper_gt[:,:,band]))
        sid.append(spectral_divergence(hyper_inf[:,:,band].reshape(-1),hyper_gt[:,:,band].reshape(-1)))
    
    avg_sam.append(sam)
    avg_sid.append(sid)
    avg_ssim.append(ssim)
    avg_psnr.append(psnr)
    avg_mrae.append(mrae)
    avg_rmse.append(rmse)
    
    #sid = test_sid(hyper_inf,hyper_gt)
    #ssim = test_ssim(hyper_inf,hyper_gt)
    #mrae = test_mrae(hyper_inf,hyper_gt)
    #rmse = test_rmse(hyper_inf,hyper_gt)
    #psnr = test_psnr(hyper_inf,hyper_gt)
    #print(sam)
    #avg_mrae.update(mrae)
    #avg_rmse.update(rmse)
    #avg_psnr.update(psnr)
    #avg_sam.update(sam)
    #avg_sid.update(sid)
    #avg_ssim.update(ssim)

    #print(f'mrae:{mrae}, rmse:{rmse}, sam: {sam}, sid: {sid}, ssim: {ssim}, psnr:{psnr}')

avg_sam = np.float32(avg_sam)
avg_sam = np.mean(avg_sam, axis=0)
print(avg_sam.shape)
avg_sid = np.float32(avg_sid)
avg_sid = np.mean(avg_sid, axis=0)
print(avg_sid.shape)
avg_ssim = np.float32(avg_ssim)
avg_ssim = np.mean(avg_ssim, axis=0)
print(avg_ssim.shape)
avg_psnr = np.float32(avg_psnr)
avg_psnr = np.mean(avg_psnr, axis=0)
print(avg_psnr.shape)
avg_mrae = np.float32(avg_mrae)
avg_mrae = np.mean(avg_mrae, axis=0)
print(avg_mrae.shape)
avg_rmse = np.float32(avg_rmse)
avg_rmse = np.mean(avg_rmse, axis=0)
print(avg_rmse.shape)

final = np.dstack((avg_mrae, avg_rmse, avg_sam, avg_sid, avg_ssim, avg_psnr))
print(final.shape)
save_matv73('rgbn', 'error', final)
#print(f'Final:RGB-NIR sensor, mrae:{avg_mrae.avg}, rmse:{avg_rmse.avg}, sam: {avg_sam.avg}, sid: {avg_sid.avg}, ssim: {avg_ssim.avg}, psnr:{avg_psnr.avg}')


