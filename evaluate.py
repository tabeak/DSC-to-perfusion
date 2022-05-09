import numpy as np
import time
import os
import matplotlib
import argparse
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import TensorDataset, DataLoader

import config as c
from utils import eval_utils as eut


matplotlib.use("Agg")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--database',
    type=str,
    help="Type hdb, 1kplus or all",  # all not supported yet
    required=True
)
parser.add_argument(
    '--dataset',
    type=str,
    help="Type train for training set, test for test set or val for validation set",
    required=True
)
parser.add_argument(
    '--trial',
    type=str,
    help="Select trial that should be evaluated",
    required=True
)
parser.add_argument(
    '--epoch',
    type=str,
    help="Select epoch that should be evaluated or type best for epoch with best validation loss",
    required=True
)
parser.add_argument(
    '--netG_type',
    type=str,
    help="Specify type of generator, e.g. tmpandunet, stunet, etc",
    required=True
)
parser.add_argument(
    "--upsampling", type=bool, default=False, help="Usage of upsampling layer in G"
)
parser.add_argument(
    "--seed", type=int, default=12, help="Set random seed, default: 12"
)
parser.add_argument(
    "--norm_layer_g", type=str, default="batchnorm", 
    help="Type of normalization, batchnorm, instancenorm or spectralnorm"
)
parser.add_argument("--nr_layer_g", type=int, default=7, 
    help="number of layers G (Unet part)")
parser.add_argument("--ngf", type=int, default=64, help="number of filters G")
FLAGS, _ = parser.parse_known_args()
database = FLAGS.database
dataset = FLAGS.dataset
trial = FLAGS.trial
sel_e = FLAGS.epoch
netG_type = FLAGS.netG_type
ngf = FLAGS.ngf
normlayer = FLAGS.norm_layer_g
nr_layer_g = FLAGS.nr_layer_g

if c.use_gpu and not torch.cuda.is_available():
    raise Exception("No GPU found, please change use_gpu to False")

if c.use_gpu:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    os.environ["PYTHONHASHSEED"] = str(FLAGS.seed)
    Tensor = torch.cuda.FloatTensor
    device = torch.device("cuda:0")
else:
    torch.manual_seed(FLAGS.seed)
    device = torch.device("cpu")

# get best epoch for val loss
if sel_e=="best":
    losses = np.load("{}models/DSC-perf/Trial_{}/gen_imgs/losses.npz".format(c.root_path, 
                                                                             trial))
    val_loss = losses["val"]
    sel_e = np.argmin(val_loss)

print("Loading model. Trial: {}, Epoch: {}".format(trial, sel_e))
net_g, saved_params = eut.load_g(trial, sel_e, netG_type, ngf=ngf, num_downs=nr_layer_g, 
                                 ups=FLAGS.upsampling, normlayer=normlayer)
net_g = net_g.to(device)
# get input and output images
input_seq = saved_params["input"]
output_seq = saved_params["output"]

print("Get all files for evaluation. Evaluatiing {} set.".format(dataset))
img_path = c.data_path + dataset
if not database=="all":
    out_files = sorted(os.listdir("{}/{}/{}".format(img_path, database, output_seq[0])))
    in_files = sorted(os.listdir("{}/{}/{}".format(img_path, database, input_seq[0])))
    print("out files:",out_files)
    print("in files:",in_files)
    assert len(in_files)==len(out_files)

if c.save_nii:
    save_path_root = "{}models/DSC-perf/Trial_{}/pred/".format(c.root_path, trial)
    save_path_nii = save_path_root + dataset + "/"
    if not os.path.isdir(save_path_nii):
        if not os.path.isdir(save_path_root):
            os.mkdir(save_path_root)
            os.mkdir(save_path_nii)
        else:
            os.mkdir(save_path_nii)

save_path = "{}models/DSC-perf/Trial_{}/results/".format(c.root_path, trial)
filename = "epoch{}_{}_{}.csv".format(sel_e, dataset, database)
if not os.path.isdir(save_path):
    os.mkdir(save_path)
if os.path.isfile(save_path + filename):
    os.remove(save_path + filename)
with open(save_path + filename, mode="w") as f:
    w = csv.writer(f, delimiter=",")
    w.writerow(["Patient", "MSE", "NRMSE", "PSNR", "SSIM", "MAE"])

start_time = time.time()
results_norm = np.empty((len(in_files), len(c.metrics)))
results = np.empty((len(in_files), len(c.metrics)))
for i in range(len(in_files)):
    img = nib.load("{}/{}/{}/{}".format(img_path, database, input_seq[0], in_files[i])).get_fdata()
    data = nib.load("{}/{}/{}/{}".format(img_path, database, output_seq[0], out_files[i]))
    print("input path", in_files[i])
    print("output path", out_files[i])
    print("input shape", img.shape)
    print("output shape", data.shape)
    if not data.shape==(128,128,21):
        data = eut.resize_img(data, (128,128,21))
    target = data.get_fdata()
    print("new output shape:", data.shape)
    if c.save_nii:
        pred = np.empty(target.shape)

    results_pat_norm = np.empty((img.shape[2], len(c.metrics)))
    # normalize
    if c.datanorm=="once":
        minmax_in = np.load("{}train_{}_{}_min_max_values.npz".format(c.data_path, 
                                                                      input_seq[0], 
                                                                      database))
        norm_img = 2 * ((img - minmax_in["min"]) / (minmax_in["max"] - minmax_in["min"])) - 1
        minmax_out = np.load("{}train_{}_{}_min_max_values.npz".format(c.data_path, 
                                                                       output_seq[0], 
                                                                       database))
        norm_target = 2 * ((target - minmax_out["min"]) / (minmax_out["max"] - minmax_out["min"])) - 1

    for sl_nr in range(img.shape[2]):
        if input_seq[0].startswith("rawDSC"):
            if database=="peg":
                input_sl = np.empty((1,80,128,128))
                input_sl = np.rot90(np.rollaxis(norm_img[:, :, sl_nr], 2, 0), 3, axes=(1,2))
                input_sl_notnorm = np.rot90(np.rollaxis(img[:, :, sl_nr], 2, 0), 3, axes=(1,2))
            elif database=="hdb":
                input_sl = np.rot90(np.rollaxis(norm_img[:, :, sl_nr], 2, 0), 1, axes=(1,2))
            input_sl = np.expand_dims(input_sl, axis=0)
        else:
            input_sl = np.expand_dims(norm_img[:,:,sl_nr], axis=0)
            input_sl = np.expand_dims(input_sl, axis=0)
        if sl_nr==0:
            print("input_sl shape:",input_sl.shape)
        
        if database=="peg":
            target_sl = np.expand_dims(np.fliplr(np.rot90(norm_target[:,:,sl_nr],1)),axis=0)
        elif database=="hdb":
            target_sl = np.expand_dims(np.rot90(norm_target[:, :, sl_nr]), axis=0)
        target_sl = np.expand_dims(target_sl, axis=0)
        if sl_nr==0:
            print("target_sl shape:",target_sl.shape)

        #print(input_sl.shape)
        pred_sl = net_g(torch.Tensor(input_sl.copy()).cuda())
        pred_sl = pred_sl.detach().cpu().numpy().astype(np.float64)
        if sl_nr==0:
            print("pred_sl shape:",pred_sl.shape)        
        if c.save_nii:
            pred[:, :, sl_nr] = pred_sl
            if sl_nr==15:
                plt.figure()
                plt.subplot(1,3,1)
                plt.imshow(input_sl[0,0,:,:])
                plt.subplot(1,3,2)
                plt.imshow(pred_sl.squeeze())
                plt.subplot(1,3,3)
                plt.imshow(target_sl.squeeze())
                plt.savefig(save_path_root + out_files[i] + ".png")
            
            
        # calculate metrics
        results_pat[sl_nr, 0] = mse(target_sl, pred_sl)
        results_pat[sl_nr, 1] = nrmse(target_sl, pred_sl)
        results_pat[sl_nr, 2] = psnr(target_sl, pred_sl, data_range=2)
        results_pat[sl_nr, 3] = ssim(target_sl.squeeze(), 
                                     pred_sl.squeeze())
        results_pat[sl_nr, 4] = eut.mae(target_sl.squeeze(), 
                                      pred_sl.squeeze())

    res_mean = np.mean(results_pat, axis=0)
    print("Performance for patient {}: {}".format(i, res_mean))

    results[i, :] = res_mean

    # save performance for one patient
    with open(save_path + filename, mode="a") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow([out_files[i], res_mean[0], res_mean[1],
                    res_mean[2], res_mean[3], res_mean[4]])

    if c.save_nii:
        # scale back to original 
        if c.datanorm=="once":
            pred_new = (minmax_out["max"]-minmax_out["min"]) * ((pred + 1) / 2) + minmax_out["min"]
            print(np.min(pred_new), np.max(pred_new))
        else:
            pred_new = (pred - np.min(pred)) * (np.max(target) - \
                       np.min(target))/(np.max(pred) - np.min(pred)) + np.min(target)
        pred_targ = (pred - np.min(pred)) * (np.max(target) - \
                     np.min(target))/(np.max(pred) - np.min(pred)) + np.min(target)
        if database=="peg":
            pred_new = np.fliplr(np.rot90(pred_new, 1))
            pred_norm = np.fliplr(np.rot90(pred, 1))
            pred_new_mask = pred_new
            if i==0:
                plt.subplot(121)
                plt.imshow(pred_new[:,:,12])
                plt.subplot(122)
                plt.imshow(target[:,:,12])
                plt.savefig("test.png")
        elif database=="hdb":
            pred_new = np.rot90(pred_new, 3)
            pred_norm = np.rot90(pred, 3)
            pred_targ = np.rot90(pred_targ, 3)
            if c.use_DSC_mask:
                pred_new_mask = pred_new
                pred_new_mask[img[:,:,:,0]<=0] = np.min(pred_new_mask)
            else:
                pred_new_mask = pred_new
        pred_nii = nib.Nifti1Image(pred_new_mask, affine=data.affine, 
                                   header=data.header)
        nib.save(pred_nii, save_path_nii + out_files[i])
        pred_nii_norm = nib.Nifti1Image(pred_norm, affine=data.affine, 
                                        header=data.header)
        nib.save(pred_nii_norm, save_path_nii + "norm_" + out_files[i])
        pred_nii_targ = nib.Nifti1Image(pred_targ, affine=data.affine, 
                                        header=data.header)
        nib.save(pred_nii_targ, save_path_nii + "target_" + out_files[i])
        print("Nifti saved.")

with open(save_path + filename, mode="a") as f:
    w = csv.writer(f, delimiter=",")
    tmp_res = np.mean(results, axis=0)
    w.writerow(["mean", tmp_res[0], tmp_res[1], tmp_res[2], 
                tmp_res[3], tmp_res[4]])

total_time = time.time() - start_time

print('Time for complete evaluation session: ',
      (total_time // (3600 * 24)), 'days',
      (total_time // 3600) % 24, 'hours',
      (total_time // 60) % 60, 'minutes',
      total_time % 60, 'seconds')
