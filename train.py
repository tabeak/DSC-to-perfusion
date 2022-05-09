import argparse
import time
import os
import csv
import json

import numpy as np
import pandas as pd
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

import config as c
from model import Discriminator, PatchDiscriminator, \
                  UnetGenerator, ResnetGenerator, \
                  STUnet, TmpAndUnet, TwoPathsUnet
from model3D import PatchDiscriminator3D, TmpAndUnet3D
from utils import training_utils as tut
from utils import eval_utils as eut
from utils import convNd


# process input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--trial", type=str, required=True, help="Trial number")
parser.add_argument(
    "--netG_type", type=str, required=True, help="Architecture of generator"
)
parser.add_argument(
    "--netD_type", type=str, default="patch", 
    help="Architecture of discriminator: patch or regular"
)
parser.add_argument(
    "--data",  nargs='+',type=str, default=["hdb"], 
    help="Data source: a list of hdb for Heidelberg data and/or peg for PEGASUS"
)
parser.add_argument(
    "--inputseq", nargs='+', type=str, default=["DSC"], 
    help="Input sequence, default: DSC"
)
parser.add_argument(
    "--outputseq", nargs='+', type=str, default=["TMAX"], 
    help="Output sequence: a list consisting of TMAX, TTP, CBF, CBV or MTT"
)
parser.add_argument("--batch_size", type=int, default=4, help="input batch size")
parser.add_argument("--ngf", type=int, default=64, help="number of filters G")
parser.add_argument("--ndf", type=int, default=64, help="number of filters D")
parser.add_argument("--nr_layer_g", type=int, default=7, 
    help="number of layers G (Unet part)")
parser.add_argument("--nr_layer_d", type=int, default=3, 
    help="number of layers D (for patchD)")
parser.add_argument(
    "--norm_layer_g", type=str, default="batchnorm", 
    help="Type of normalization, batchnorm, instancenorm or spectralnorm"
)
parser.add_argument("--norm_layer_d", type=str, default="batchnorm", 
    help="Type of normalization, batchnorm, instancenorm or spectralnorm"
)
parser.add_argument(
    "--weight_init", type=str, default="normal", 
    help="Weight initilization: normal, he or xavier"
)
parser.add_argument(
    "--epochs", type=int, default=200, help="number of epochs"
)
parser.add_argument(
    "--lrg", type=float, default=0.0001, help="learning rate for G, default=0.0001"
)
parser.add_argument(
    "--lrd", type=float, default=0.0001, help="learning rate for D, default=0.0001"
)
parser.add_argument(
    "--betas_g", nargs='+', type=float, default=[0.5, 0.999],
    help="betas for G, default=[0.5, 0.999]"
)
parser.add_argument(
    "--betas_d", nargs='+', type=float, default=[0.5, 0.999], 
    help="betas for D, default=[0.5, 0.999]"
)
parser.add_argument(
    "--loss_g", type=str, default="BCE", help="generator's loss, default=BCE"
)
parser.add_argument(
    "--loss_reconstr", type=str, default="l1", help="generator's reconstruction loss, default=l1"
)
parser.add_argument(
    "--loss_d", type=str, default="BCE", help="discriminator's loss, default=BCE"
)
parser.add_argument(
    "--loss_ratio", type=int, default=1, 
    help="loss ratio of adv and reconstruction loss for G, default=None"
)
parser.add_argument(
    "--n_discr", type=int, default=1, help="Number of D updates per epoch"
)
parser.add_argument(
    "--upsampling", type=bool, default=False, help="Usage of upsampling layer in G"
)
parser.add_argument(
    "--seed", type=int, default=12, help="Set random seed, default: 12"
)
opt = parser.parse_args()

if c.use_gpu and not torch.cuda.is_available():
    raise Exception("No GPU found, please change use_gpu to False")

if c.use_gpu:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    os.environ["PYTHONHASHSEED"] = str(opt.seed)
    Tensor = torch.FloatTensor
    device = torch.device("cuda:{}".format(c.gpu_idx[0]))
else:
    torch.manual_seed(opt.seed)
    #Tensor = torch.FloatTensor
    device = torch.device("cpu")

# check if output folders exists, if not create them
model_path = c.root_path + "models/DSC-perf/Trial_" + opt.trial + "/"
model_path_pretrained = c.root_path + "models/DSC-perf/Trial_" + c.load_from_trial + "/"
gen_img_path = c.root_path + "models/DSC-perf/Trial_" + opt.trial + "/gen_imgs/"
if not os.path.isdir(gen_img_path):
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        os.mkdir(gen_img_path)
    else:
        os.mkdir(gen_img_path)

# copy config, save current opt
shutil.copyfile("config.py", "{}models/DSC-perf/Trial_{}/gen_imgs/config.py".format(c.root_path, opt.trial))
f_opt = open("{}models/DSC-perf/Trial_{}/gen_imgs/opt.json".format(c.root_path, opt.trial), "w")
json.dump(opt.__dict__, f_opt)
f_opt.close()

print('===> Loading data')
train_dataloader, input_ch, output_ch = tut.load_dataset(opt.data, opt.inputseq, 
                                                         opt.outputseq, 
                                                         opt.batch_size, "train")
val_dataloader, input_ch, output_ch = tut.load_dataset(opt.data, opt.inputseq, 
                                                       opt.outputseq, opt.batch_size, 
                                                       "val", False)

if c.use_gpu:
    Tensor = torch.cuda.FloatTensor

if opt.norm_layer_g=="batchnorm":
    norm_layer_g_2d = nn.BatchNorm2d
    norm_layer_g_3d = nn.BatchNorm3d
elif opt.norm_layer_g=="instancenorm":
    norm_layer_g_2d = nn.InstanceNorm2d
    norm_layer_g_3d = nn.InstanceNorm3d
if opt.norm_layer_d=="batchnorm":
    norm_layer_d_2d = nn.BatchNorm2d
elif opt.norm_layer_d=="instancenorm":
    norm_layer_d_2d = nn.InstanceNorm2d
        
print('===> Building models')
if opt.netG_type=="unet":
    net_g = UnetGenerator(input_nc=input_ch, output_nc=output_ch, 
                          ngf=opt.ngf, norm_layer=norm_layer_g_2d, 
                          ups=opt.upsampling, num_downs=opt.nr_layer_g).to(device)
elif opt.netG_type=="tmpandunet":
    net_g = TmpAndUnet(input_nc=input_ch, output_nc=output_ch, 
                       ngf=opt.ngf, norm_layer_2d=norm_layer_g_2d, 
                       norm_layer_3d=norm_layer_g_3d, 
                       ups=opt.upsampling, nr_layer_g=opt.nr_layer_g).to(device)

if opt.netD_type=="patch":
    net_d = PatchDiscriminator(input_nc=input_ch+output_ch, ndf=opt.ndf, 
                               norm_layer=norm_layer_d_2d, 
                               n_layers=opt.nr_layer_d).to(device)
else:
    net_d = Discriminator(input_nc=input_ch+output_ch, ndf=opt.ndf, 
                          norm_layer=norm_layer_d_2d).to(device)

# initalize weights or load model
if c.continue_training:
    saved_model_path = model_path_pretrained + "epoch" + str(c.load_from_epoch) + ".pth"
    saved_params_dict = torch.load(saved_model_path)
else:
    if opt.weight_init=="normal":
        net_g.apply(tut.weights_init_normal)
        net_d.apply(tut.weights_init_normal)
    elif opt.weight_init=="xavier":
        net_g.apply(tut.weights_init_xavier)
        net_d.apply(tut.weights_init_xavier)
    if opt.weight_init=="he":
        net_g.apply(tut.weights_init_he)
        net_d.apply(tut.weights_init_he)
if c.pretrainedG:
    pretrained_dict = torch.load(c.pretrained_path, map_location=device)
    model_dict = net_g.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
    model_dict.update(pretrained_dict) 
    net_g.load_state_dict(pretrained_dict, strict=False)
      
# optimizer  
optim_g = optim.Adam(net_g.parameters(), lr=opt.lrg, betas=tuple(opt.betas_g), 
                     weight_decay=0)
if c.optimizer_d=="SGD":
    optim_d = optim.SGD(net_d.parameters(), lr=opt.lrd, momentum=0.9)
else:
    optim_d = optim.Adam(net_d.parameters(), lr=opt.lrd,
                         betas=tuple(opt.betas_d))
                         
if c.continue_training:
    # initalize weights with saved ones
    net_g.load_state_dict(saved_params_dict["generator_state_dict"])
    optim_g.load_state_dict(saved_params_dict["gen_opt_state_dict"])
    net_d.load_state_dict(saved_params_dict["discriminator_state_dict"])
    optim_d.load_state_dict(saved_params_dict["discr_opt_state_dict"])
    losses_d = []
    losses_g = []
    losses_val = []
    start_epoch = 0
else:
    losses_d = []
    losses_g = []
    losses_val = []
    start_epoch = 0

print(net_g)
print(net_d)

# multiple gpu usage
if c.use_gpu and (c.nr_gpus > 1):
    print("Let's use", c.nr_gpus, "GPUs!")
    net_g = nn.DataParallel(net_g, device_ids=c.gpu_idx)
    net_d = nn.DataParallel(net_d, device_ids=c.gpu_idx)

if (c.generate_while_train and not ("3D" in opt.netG_type)):
    plt_imgs = np.load(c.data_path + "img_for_plt_new.npz")
    output_tmp = plt_imgs[opt.outputseq[0]]
    input_tmp = plt_imgs[opt.inputseq[0]]
    input_img_plt = input_tmp[:c.nr_imgs_gen]
    input_img_gpu = Tensor(input_img_plt).to(device)
    target_img_plt = output_tmp[:c.nr_imgs_gen]

print('===> Starting training')

start_time = time.time()

# check number of parameters
nr_params_g = sum(p.numel() for p in net_g.parameters())
nr_params_d = sum(p.numel() for p in net_d.parameters())
print("nr params g:", nr_params_g)
print("nr params d:", nr_params_d)

for epoch in range(start_epoch, opt.epochs):

    print(epoch + 1, "/", opt.epochs)

    start_time_epoch = time.time()

    for i, img in enumerate(train_dataloader, 0):

        input_img = img[0].to(device)
        target_img = img[1].to(device)
        if (i==0) and (epoch==0):
            print("Input shape", input_img.shape)
            print("Target shape:", target_img.shape)
            if "3D" in opt.netG_type:
                # check if images look correct
                input_tmp = input_img.cpu().detach().numpy()
                target_tmp = target_img.cpu().detach().numpy()
                plt.subplot(1,2,1)
                plt.imshow(input_tmp[0,0,:,:,18])
                plt.subplot(1,2,2)
                plt.imshow(target_tmp[0,0,:,:,18])
                plt.savefig("input_output.png")

        ##############################
        ######### Update D ###########
        ##############################

        for _ in range(opt.n_discr):
            # get real pair
            real_pair = torch.cat((input_img, target_img), 1)
            real_pair_gpu = real_pair.to(device)

            # get generated pair
            gen_img = net_g(input_img)
            # check if NaNs are generated
            if np.isnan(gen_img.cpu().detach().numpy()).any():
                raise Exception("NaN values detected")
            gen_pair = torch.cat((input_img, gen_img), 1)
            gen_pair_gpu = gen_pair.to(device)

            b_size = real_pair_gpu.size(0)

            # get predictions
            out_real, lastconv_out_real = net_d(real_pair_gpu)
            out_gen, lastconv_out_gen = net_d(gen_pair_gpu)

            label_real = torch.ones(out_real.size()).to(device)
            label_gen = torch.zeros(out_gen.size()).to(device)

            # calculate losses
            if opt.netD_type=="patch":
                loss_d = tut.calculate_loss_d(net_d, optim_d, out_real,
                                              label_real, out_gen,
                                              label_gen, opt.loss_d, True)
            else:
                loss_d = tut.calculate_loss_d(net_d, optim_d, out_real,
                                              label_real, out_gen,
                                              label_gen, opt.loss_d, False)
                                

        ##############################
        ######### Update G ###########
        ##############################

        # forward pass D
        out_gen_new, lastconv_out_gen_new = net_d(gen_pair_gpu)
        label_gen_new = torch.ones(out_gen_new.size()).to(device)
       
        if c.feature_matching:
            out_real_new, lastconv_out_real_new = net_d(real_pair_gpu)
            adv_loss = F.l1_loss(lastconv_out_gen_new, lastconv_out_real_new)
        else:
            if opt.loss_g=="l1":
                adv_loss = F.l1_loss(out_gen_new, label_gen_new)
            elif opt.loss_g=="l2":
                criterion = nn.MSELoss()
                adv_loss = criterion(out_gen_new, label_gen_new)
            elif opt.loss_g=="hinge":
                criterion = nn.HingeEmbeddingLoss()
                adv_loss = criterion(out_gen_new, label_gen_new)
            else:
                if opt.netD_type=="patch":
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.BCELoss()
                adv_loss = criterion(out_gen_new, label_gen_new)
        if opt.loss_ratio is None:
            net_g.zero_grad()
            adv_loss.backward(retain_graph=True)
            if "perfmask" in opt.outputseq:
                reconstr_loss = F.l1_loss(gen_img, target_img)        
            else:
                reconstr_loss = F.l1_loss(gen_img, target_img)        
            reconstr_loss.backward(retain_graph=True)
            optim_g.step()
            loss_g_new = adv_loss.detach() + reconstr_loss.detach()           
        else:
            if "perfmask" in opt.outputseq:
                if opt.loss_reconstr=="l1":
                    reconstr_loss = F.l1_loss(gen_img, target_img)
                elif opt.loss_reconstr=="l2":
                    criterion = nn.MSELoss()
                    reconstr_loss = criterion(gen_img, target_img)
                elif opt.loss_reconstr=="dice":
                    reconstr_loss = tut.dice_loss(gen_img, target_img)
                elif opt.loss_reconstr=="l1dice":
                    reconstr_loss = F.l1_loss(gen_img, target_img) + tut.dice_loss(gen_img, target_img)
            else:
                reconstr_loss = F.l1_loss(gen_img, target_img)
            loss_g = opt.loss_ratio * reconstr_loss + adv_loss
            net_g.zero_grad()
            loss_g.backward()
            optim_g.step()
            loss_g_new = loss_g.detach()

    losses_d.append(loss_d.cpu())
    losses_g.append(loss_g_new.cpu())

    end_time_epoch = time.time()
    print("Training for epoch", epoch + 1, "took",
          (end_time_epoch - start_time_epoch) / 60, "minutes.")
    print("D loss:", loss_d)
    print("G loss:", loss_g_new)
    print("adv loss:", adv_loss) 
    mselist, nrmselist, psnrlist, ssimlist, maelist = [], [], [], [], []
    with torch.no_grad():   
        val_loss = 0
        for i, img in enumerate(val_dataloader, 0):
            input_img = img[0].to(device)
            target_img = img[1].to(device)
            for j in range(input_img.shape[0]):
                if "3D" in opt.netG_type:
                    gen_img = net_g(input_img)
                    target_img_tmp = target_img
                else:
                    gen_img = net_g(torch.unsqueeze(input_img[j],0))
                    target_img_tmp = torch.unsqueeze(target_img[j], 0)
                target_img_np = target_img_tmp.cpu().detach().numpy().astype(np.float64)
                val_loss += F.l1_loss(gen_img, target_img_tmp)

                gen_img_np = gen_img.cpu().detach().numpy().astype(np.float64)
                # get metrics
                mselist.append(mse(target_img_np, gen_img_np))
                nrmselist.append(nrmse(target_img_np, gen_img_np))
                if (c.datanorm=="meanvar") or (c.datanorm=="meansd") or (c.datanorm=="perc"):
                    psnrlist.append(psnr(target_img_np, gen_img_np, data_range=np.max(target_img_np)-np.min(target_img_np)))
                else:
                    psnrlist.append(psnr(target_img_np, gen_img_np,data_range=2))
                if opt.batch_size<8:
                    nr_img = target_img_np.shape[0]
                    tmp_list = np.empty((nr_img))
                    for d in range(nr_img):
                        tmp_list[d] = ssim(target_img_np[d].squeeze(), gen_img_np[d].squeeze())
                    ssimlist.append(np.sum(tmp_list)/nr_img)
                else:
                    ssimlist.append(ssim(target_img_np.squeeze(), gen_img_np.squeeze()))
                maelist.append(eut.mae(target_img_np.squeeze(), gen_img_np.squeeze()))
        losses_val.append(val_loss.detach().cpu())
        print("Val loss:", val_loss.detach())

    # save model
    if c.save_model or (epoch==(opt.epochs-1)):
        if (epoch in np.arange(start_epoch-1, opt.epochs, c.save_every_X_epoch)) or (epoch==(opt.epochs-1)):
            tut.save_model(trial_nr=opt.trial, epoch=epoch, net_g=net_g, net_d=net_d, 
                           optim_g=optim_g, optim_d=optim_d, losses_g=losses_g, 
                           losses_d=losses_d, batch_size=opt.batch_size, 
                           lr_d=opt.lrd, lr_g=opt.lrg, input_seq=opt.inputseq, 
                           output_seq=opt.outputseq, betas_d=opt.betas_d, betas_g=opt.betas_g,
                           model_path=model_path)

    if (c.generate_while_train and not ("3D" in opt.netG_type)):
        with torch.no_grad():
            gen_img_plt = net_g(input_img_gpu)
            gen_img_plt = gen_img_plt.cpu()
            gen_img_plt = gen_img_plt.detach().numpy()

        eut.save_figs(epoch, input_img_plt, target_img_plt, gen_img_plt, 
                      losses_g, losses_d, losses_val, gen_img_path)
    elif opt.netG_type=="tmpandunet3D":
        input_img_plt = input_img[0,0,:,:,12].cpu().detach().numpy()
        target_img_plt = target_img_np[0,0,:,:,12]
        gen_img_plt = gen_img_np[0,0,:,:,12]
        eut.save_figs(epoch, input_img_plt, target_img_plt, gen_img_plt, 
                      losses_g, losses_d, losses_val, gen_img_path, True)

    save_path = "{}models/DSC-perf/Trial_{}/gen_imgs/results.csv".format(c.root_path, opt.trial)
    if not os.path.isfile(save_path):
        with open(save_path, mode="w") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["epoch", "MSE", "NRMSE", "PSNR", "SSIM", "MAE"])
    
    with open(save_path, mode="a") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow([epoch, np.mean(mselist), np.mean(nrmselist), 
                    np.mean(psnrlist), np.mean(ssimlist), 
                    np.mean(maelist)])
        

# get best epochs for each metric
df = pd.read_csv(save_path)
print("Best MSE of {} in epoch {}".format(df["MSE"].min(), df.loc[df["MSE"].idxmin()][0]))
print("Best NRMSE of {} in epoch {}".format(df["NRMSE"].min(), df.loc[df["NRMSE"].idxmin()][0]))
print("Best PSNR of {} in epoch {}".format(df["PSNR"].max(), df.loc[df["PSNR"].idxmax()][0]))
print("Best SSIM of {} in epoch {}".format(df["SSIM"].max(), df.loc[df["SSIM"].idxmax()][0]))
print("Best MAE of {} in epoch {}".format(df["MAE"].min(), df.loc[df["MAE"].idxmin()][0]))

