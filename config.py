# set path depending on machine
cluster = "research"
if cluster=="clinic":
    root_path = "/data/users/kossent/work/tabea/"
elif cluster=="research":
    root_path = "/fast/users/kossent_c/work/tabea/"
elif cluster=="DL2":
    root_path = "/data-nvme/tabea/"


data_path = root_path + "data/DSC-perf/"
datanorm = "once"  # normalization scheme
pretrainedG = False 
# path for pretrained model 
pretrained_path = root_path + "models/pretrained/maps2sat_G.pth"  
save_model = True
save_every_X_epoch = 10
generate_while_train = True
nr_imgs_gen = 4  # how many images are generated while training
# training settings
dropout_rate = 0  # 0.1

use_gpu = True
gpu_idx = [0]
nr_gpus = len(gpu_idx)
bn = True
threads = 4  # for loading data

# optimizers
optimizer_d = "adam"

# only for continued training
continue_training = True
load_from_epoch = 99
load_from_trial = "pix2pix"

# for evaluation
metrics = ["MSE", "NRMSE", "PSNR", "SSIM", "MAE"]
save_nii = True
use_DSC_mask = False

