import numpy as np
import os
import torch

import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import nibabel as nib

import config as c
from utils import pytorch_ssim 


def load_dataset(data, input_seq, output_seq, batch_size=4, datasplit="train", shffl=True):
    Tensor = torch.FloatTensor
    # INPUT
    for i in range(len(data)):
        for j in range(len(input_seq)):
            input_path = "{}{}_{}_{}_norm{}_all.npz".format(c.data_path, datasplit, 
                                                          input_seq[j], data[i], 
                                                          c.datanorm)
            input_data = np.load(input_path)
            input_img = Tensor(input_data["imgs"])
            #if not input_seq[j].startswith("raw"):
            #    input_img = F.pad(input_img, (4, 4))
            if j==0:
                tmp_input = input_img
            else:
                tmp_input = torch.cat((tmp_input, input_data), 1)
        if i==0:
            final_input = tmp_input
        else:
            final_input = torch.cat((final_input, tmp_input), 0)
    print("Shape of {} input data: {}".format(datasplit, final_input.shape))
    # OUTPUT
    for i in range(len(data)):
        for j in range(len(output_seq)):
            output_path = "{}{}_{}_{}_norm{}_all.npz".format(c.data_path, datasplit, 
            #output_path = "{}{}_{}_{}_all.npz".format(c.data_path, datasplit, 
                                                           output_seq[j], data[i],
                                                           c.datanorm)
            output_data = np.load(output_path)
            output_img = Tensor(output_data["imgs"])
            #if not output_seq[j].startswith("raw"):
            #    output_img = F.pad(output_img, (4, 4))
            if j==0:
                tmp_output = output_img
            else:
                tmp_output = torch.cat((tmp_output, output_data), 1)
        if i==0:
            final_output = tmp_output
        else:
            final_output = torch.cat((final_output, tmp_output), 0)
    print("Shape of {} output data: {}".format(datasplit, final_output.shape))
    # put together
    dataset = TensorDataset(final_input, final_output)
    dataloader = DataLoader(dataset=dataset, num_workers=c.threads,
                            batch_size=batch_size, shuffle=shffl)
    return dataloader, final_input.shape[1], final_output.shape[1]
    

def weights_init_normal(m, init_mean=0, init_sd=0.05):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, init_mean, init_sd)
        if m.bias is not None:
            nn.init.normal_(m.bias, init_mean, init_sd)
    classname = m.__class__.__name__
    if (classname.find("BatchNorm2d") != -1) or (classname.find("BatchNorm3d") != -1):
        nn.init.normal_(m.weight, 0.0, 0.05)
        nn.init.constant_(m.bias, 0)


def weights_init_xavier(m, init_mean=0, init_sd=0.05):
    if isinstance(m, nn.Conv2d):
        print("Xavier weight initilization")
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias, init_mean, init_sd)
    classname = m.__class__.__name__
    if (classname.find("BatchNorm2d") != -1) or (classname.find("BatchNorm3d") != -1):
        nn.init.normal_(m.weight, 0.0, 0.05)
        nn.init.constant_(m.bias, 0)


def weights_init_he(m, init_mean=0, init_sd=0.05):
    if isinstance(m, nn.Conv2d):
        print("He weight initialization")
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.normal_(m.bias, init_mean, init_sd)
    classname = m.__class__.__name__
    if (classname.find("BatchNorm2d") != -1) or (classname.find("BatchNorm3d") != -1):
        nn.init.normal_(m.weight, 0.0, 0.05)
        nn.init.constant_(m.bias, 0)


def calculate_loss_d(net_d, optim_d, out_real, label_real, out_gen, label_gen, 
                     loss_d, patchD):
    if loss_d == "l1":
        real_loss = 0.5 * F.l1_loss(out_real, label_real)
    elif loss_d == "l2":
        criterion = nn.MSELoss()
        real_loss = 0.5 * criterion(out_real, label_real)
    elif loss_d == "hinge":
        criterion = nn.HingeEmbeddingLoss()
        real_loss = 0.5 * criterion(out_real, label_real)
    elif loss_d == "ssim":
        criterion = pytorch_ssim.SSIM()
        real_loss = 1 - criterion(out_real, label_real)
    else:
        if patchD:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.BCELoss()
        real_loss = 0.5 * criterion(out_real, label_real)
    net_d.zero_grad()
    real_loss.backward(retain_graph=True)

    if loss_d == "l1":
        gen_loss = 0.5 * F.l1_loss(out_gen, label_gen)
    else:
        gen_loss = 0.5 * criterion(out_gen, label_gen)

    gen_loss.backward(retain_graph=True)

    optim_d.step()
    # save loss of discriminator
    loss_d = (real_loss.detach() + gen_loss.detach()) / 2

    return loss_d


def save_model(trial_nr, epoch, net_g, net_d, optim_g, optim_d, losses_g, losses_d, 
               batch_size, lr_d, lr_g, input_seq, output_seq, betas_g, betas_d, model_path):
    model_path = model_path + "epoch" + str(epoch) + ".pth"
    torch.save({
        "trial_nr": trial_nr,
        "input": input_seq,
        "output": output_seq,
        "epoch": epoch,
        "lr_d": lr_d,
        "lr_g": lr_g,
        "beta1_d": betas_d[0],
        "beta2_d": betas_d[1],
        "beta1_g": betas_g[0],
        "beta2_g": betas_d[1],
        "device": c.gpu_idx,
        "batch_size": batch_size,
        "generator_state_dict": net_g.state_dict(),
        "discriminator_state_dict": net_d.state_dict(),
        "gen_opt_state_dict": optim_g.state_dict(),
        "discr_opt_state_dict": optim_d.state_dict(),
        "generator_loss": losses_g,
        "discriminator_loss": losses_d
    }, model_path)


def normalize(x):
    return 2 * ((x - torch.min(x)) / (torch.max(x) - torch.min(x))) - 1


def dice_loss(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))
              
