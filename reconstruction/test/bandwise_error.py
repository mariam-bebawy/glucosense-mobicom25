import numpy as np
import hdf5storage
import matplotlib
import matplotlib.pyplot as plt
from utils import AverageMeter, save_matv73
from spectral_metrics import test_msam, test_sid, test_ssim, test_mrae, test_rmse, test_psnr, spectral_angle, spectral_divergence

data_root = '../../../datasets/working_glucose/Hyperspectral/'
hyper_gt_path = f'{data_root}/HS_GT/'
hyper_inf_path = f'{data_root}/reconstructed/'

with open(f'{data_root}/test.txt', 'r') as fin:
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


print(f'mrae:{avg_mrae}, rmse:{avg_rmse}, sam: {avg_sam}, sid: {avg_sid}, ssim: {avg_ssim}, psnr:{avg_psnr}')

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
save_matv73('rgb_nir', 'error', final)
#print(f'Final:RGB-NIR sensor, mrae:{avg_mrae.avg}, rmse:{avg_rmse.avg}, sam: {avg_sam.avg}, sid: {avg_sid.avg}, ssim: {avg_ssim.avg}, psnr:{avg_psnr.avg}')

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 11))
x=range(400, 1004, 9)
axs[0,0].plot(x,avg_mrae,linewidth=2,label="MRAE")
axs[0,1].plot(x,avg_rmse,linewidth=2,label="RMSE")
axs[0,2].plot(x,avg_sam,linewidth=2,label="SAM")
axs[1,0].plot(x,avg_sid,linewidth=2,label="SID")
axs[1,1].plot(x,avg_ssim,linewidth=2,label="SSIM")
axs[1,2].plot(x,avg_psnr,linewidth=2,label="PSNR")
axs[0,0].set_xlabel('Wavelength (nm)')
axs[0,0].set_ylabel('MRAE')
axs[0,0].set_xlim([400,1000])
axs[0,0].set_ylim([0,0.5])
axs[0,1].set_xlabel('Wavelength (nm)')
axs[0,1].set_ylabel('RMSE')
axs[0,1].set_xlim([400,1000])
axs[0,1].set_ylim([0,0.5])
axs[0,2].set_xlabel('Wavelength (nm)')
axs[0,2].set_ylabel('SAM')
axs[0,2].set_xlim([400,1000])
axs[0,2].set_ylim([0,0.5])
axs[1,0].set_xlabel('Wavelength (nm)')
axs[1,0].set_ylabel('SID')
axs[1,0].set_xlim([400,1000])
axs[1,0].set_ylim([0,0.1])
axs[1,1].set_xlabel('Wavelength (nm)')
axs[1,1].set_ylabel('SSIM')
axs[1,1].set_xlim([400,1000])
axs[1,1].set_ylim([0,1])
axs[1,2].set_xlabel('Wavelength (nm)')
axs[1,2].set_ylabel('PSNR')
axs[1,2].set_xlim([400,1000])
axs[1,2].set_ylim([20,65])
plt.tight_layout()
plt.show()

for j in range(axs.shape[1]):
        img = inf_file[:,:,VIEW_BANDS[j]].reshape(512, 512)
        axs[0, j].imshow(exposure.adjust_gamma(img, 0.25), interpolation="nearest", cmap="gray")
        axs[0, j].set_title(str(ACTUAL_BANDS[j]) + " nm", **title_font_dict)
        #axs[0, j].set_xticks([])
        #axs[0, j].set_yticks([])

        # Ground Truth Hypercube (gamma adjustment)
        lab = gt_file[:,:,VIEW_BANDS[j]].reshape(512, 512)
        axs[1, j].imshow(exposure.adjust_gamma(lab, 0.25), interpolation="nearest", cmap="gray")
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])

        # Difference b/w the two hypercubes
        # diff = np.abs(lab - img)
        cfl = cfl_file[:,:,VIEW_BANDS[j]].reshape(512, 512)
        axs[2, j].imshow(exposure.adjust_gamma(cfl, 0.25), interpolation="nearest", cmap="gray")
        axs[2, j].text(75, 570, " ", **text_font_dict)
        axs[2, j].set_xticks([])
        axs[2, j].set_yticks([])

# norm = matplotlib.colors.Normalize(0, 1)
# divider = make_axes_locatable(plt.gca())
# cax = divider.append_axes("right", "5%", pad="1%")
# cax = fig.add_axes([0.945, 0.0455, 0.015, 0.2945])
# cax.tick_params(labelsize=18)
# matplotlib.colorbar.ColorbarBase(cax, cmap=plt.get_cmap("hot_r"), norm=norm)
axs[0, len(VIEW_BANDS) - 1].text(525, 425, "MobiSpectral", rotation=-90, **text_font_dict)
axs[1, len(VIEW_BANDS) - 1].text(525, 350, "Halogen", rotation=-90, **text_font_dict)
axs[2, len(VIEW_BANDS) - 1].text(525, 275, "CFL", rotation=-90, **text_font_dict)
fig.tight_layout(pad=1, h_pad=1, w_pad=-5)
fig.savefig(os.path.join(PLOTS_PATH, MODEL_NAME, "%s.pdf" % (gt_filename)), dpi=fig.dpi*2, bbox_inches="tight")
plt.show()
