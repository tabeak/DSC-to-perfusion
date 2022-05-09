# Image-To-Image Generative Adversarial Networks for Synthesizing Perfusion Parameter Maps from DSC-MR Images In Cerebrovascular Disease

## Aim

The aim of the project was to synthesize perfusion parameter maps from dynamic susceptibility contrast magnetic resonance imaging. For this, we implemented a pix2pix GAN as well as a pix2pix GAN with a temporal component, coined the temp-pix2pix GAN.

## Files

* [config.py](config.py): configuration file 
* [train.py](train.py): script for running both the pix2pix and temp-pix2pix GAN
* [model.py](model.py): GAN model file
* [evaluate.py](evaluate.py): script for evaluating the generated perfusion maps
* [utils](utils): folder containing all helper functions 
* [dsc-perf.yml](dsc-perf.yml): conda environment to be able to run all scripts

## How to run

1. Install the conda environment using the dsc-perf.yml file
2. Specify configurations and paths in config.py
3. To run the GAN with differential privacy, specify the hyperparameters, e.g.:
    ```python
    python3 train.py \
        --trial "GAN1" \               # trial name
        --data dataset1 \              # name of dataset
        --netG_type tmpandunet \       # Architecture of generator (tmpandunet or unet)
        --netD_type patch \            # Architecture of discriminator
        --inputseq DSC \               # input for G
        --outputseq Tmax \             # output for G
        --batch_size \                 # batch size
        --ngf 128 --ndf 64 \           # number of filters G and D
        --nr_layer_g 6 --nr_layer_d 2  # number of layers G and D
        --norm_layer_g batchnorm \     # type of normalization (G)
        --norm_layer_d batchnorm \     # type of normalization (D)
        --weight_init normal \         # weight initialization distribution
        --epochs 100 \                 # number of epochs
        --lrg 0.0001 --lrd 0.0001 \    # learning rate D and G
        --betas_g 0.5 0.999 \          # optimization parameters G
        --betas_d 0.5 0.999 \          # optimization parameters D
        --loss_g BCE --loss_d BCE \    # loss G and D
        --loss_reconstr l1 \           # reconstruction loss
        --loss_ratio \                 # ratio between adversarial and reconstruction loss for G
        --n_discr 1 \                  # number of D updates
        --upsampling True \            # Upsampling vs deconvolutional layer
        --seed 12                      # number for random seed                 
    ```
4. For generating the perfusion maps and evaluate them, run evaluate.py e.g.:
    ```python
    python3 evaluate.py \
        --trial "GAN1"                 # trial name 
        --database dataset1 \          # name of dataset
        --dataset test \               # dataset split
        --epoch 99  \                  # determine epoch for generation
        --netG_type \                  # architecture of G
        --upsampling True \            # Upsampling vs deconvolutional layer
        --norm_layer_g batchnorm \     # type of normalization (G)
        --nr_layer_g 6 \               # number of layers G
        --ngf 128 \                    # number of filters G
        --seed 12                      # number for random seed
    ```

## How to cite

*The preprint for this study will soon be available. As soon as the paper is published, this will be updated*
