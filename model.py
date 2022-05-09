import math
import torch
import functools
import torch.nn as nn
from torch.nn.modules.utils import _triple

import config as c


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Discriminator(nn.Module):

    def __init__(self, input_nc=2, dropout_rate=c.dropout_rate, ndf=64, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()

        self.dropout_rate = dropout_rate
        self.ch = ndf
        self.pool = 2
        self.s = 1
        self.k = 3
        self.padding = 1


        def discr_block(in_ch, out_ch, k=self.k, p=self.pool, s=self.s,
                        pad=self.padding):
            block1 = [nn.Conv2d(in_channels=in_ch, out_channels=out_ch,
                                kernel_size=k, stride=s, padding=pad),
                        nn.LeakyReLU(0.2, inplace=True),
                        norm_layer(out_ch)]
            block2 = [nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                                kernel_size=k, stride=p, padding=pad),
                        nn.LeakyReLU(0.2, inplace=True),
                        norm_layer(out_ch)]
            return block1, block2

        bl1, bl2 = discr_block(input_nc, self.ch)
        self.layer1 = nn.Sequential(*bl1)
        self.layer2 = nn.Sequential(*bl2)
        bl3, bl4 = discr_block(self.ch, 2 * self.ch)
        self.layer3 = nn.Sequential(*bl3)
        self.layer4 = nn.Sequential(*bl4)
        bl5, bl6 = discr_block(2 * self.ch, 4 * self.ch)
        self.layer5 = nn.Sequential(*bl5)
        self.layer6 = nn.Sequential(*bl6)
        bl7, bl8 = discr_block(4 * self.ch, 8 * self.ch)
        self.layer7 = nn.Sequential(*bl7)
        self.layer8 = nn.Sequential(*bl8)

        if c.WGAN:
            self.adv_layer = nn.Sequential(
                nn.Conv2d(in_channels=8 * self.ch, out_channels=1,
                          kernel_size=self.k, stride=self.s,
                          padding=self.padding))
        else:
            self.adv_layer = nn.Sequential(nn.Conv2d(in_channels=8 * self.ch,
                                                        out_channels=1,
                                                        kernel_size=8,
                                                        padding=0),
                                            nn.Sigmoid())


    def forward(self, img_pairs):
        x = self.layer1(img_pairs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x_lastconv = self.layer8(x)  # shape: (b,128,6,6)

        # final layer
        if c.WGAN:
            x_out = self.adv_layer(x_lastconv)
        else:
            x_out = self.adv_layer(x_lastconv)
            x_out = torch.reshape(x_out, (x_out.shape[0], 1))

        return x_out, x_lastconv


class PatchDiscriminator(nn.Module):
    
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of input channels
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] 
        sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator for simple pix2pix GAN"""

    def __init__(self, input_nc=80, output_nc=1, num_downs=7, ngf=16, 
                 norm_layer=nn.BatchNorm2d, use_dropout=True, ups = False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. 
                               For example, # if |num_downs| == 7, image of size 
                               128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            use_dropout     -- dropout True or False
            ups (bool)      -- use Upsampling layers if true, 
                               otherwise ConvTranspose
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        self.upsampling = ups
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, 
                                             norm_layer=norm_layer, innermost=True, 
                                             ups=self.upsampling)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, 
                                                 norm_layer=norm_layer, use_dropout=use_dropout, 
                                                 ups=self.upsampling)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, 
                                             norm_layer=norm_layer, ups=self.upsampling)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, 
                                             norm_layer=norm_layer, ups=self.upsampling)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, 
                                             norm_layer=norm_layer, ups=self.upsampling)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, 
                                             outermost=True, norm_layer=norm_layer, 
                                             ups=self.upsampling)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=True, ups=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            ups (bool)     -- use Upsampling layers if true, 
                              otherwise ConvTranspose
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.upsampling = ups
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if self.upsampling:
                upconv = UpsampleConvLayer(inner_nc * 2, outer_nc, 
                                           kernel_size=3, stride=1, 
                                           upsample=2)      
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            if self.upsampling:
                upconv = UpsampleConvLayer(inner_nc, outer_nc, 
                                           kernel_size=3, stride=1, 
                                           upsample=2)
            else:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if self.upsampling:
                upconv = UpsampleConvLayer(inner_nc * 2, outer_nc,
                                           kernel_size=3, stride=1, 
                                           upsample=2)
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(c.dropout_rate)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class TmpAndUnet(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer_3d=nn.BatchNorm3d, 
                 norm_layer_2d=nn.BatchNorm2d, ups=False, ngf=64, nr_layer_g=7):
        super(TmpAndUnet, self).__init__()

        self.ch = ngf
        self.in_ch = input_nc
        self.out_ch = output_nc
        self.norm_layer_2d = norm_layer_2d
        self.norm_layer_3d = norm_layer_3d
        self.upsampling = ups

        self.pool = 2
        self.s = 2
        self.k = 4
        self.padding = 1

        def tmp_block(in_ch, out_ch, k=self.k, p=self.pool, s=self.s,
                      pad=self.padding):
            block = [nn.Conv3d(in_channels=in_ch,
                               out_channels=out_ch,
                               kernel_size=[4,1,1], stride=[2,1,1],
                               padding=[1,0,0]),
                    self.norm_layer_3d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True)]
            return block

        self.tmp_1 = nn.Sequential(*tmp_block(1, self.ch))
        self.tmp_2 = nn.Sequential(*tmp_block(self.ch, 2 * self.ch))
        self.tmp_3 = nn.Sequential(*tmp_block(2 * self.ch, 4 * self.ch))
        self.tmp_4 = nn.Sequential(*tmp_block(4 * self.ch, 8 * self.ch))
        self.tmp_5 = nn.Sequential(*tmp_block(8 * self.ch, 8 * self.ch))
        self.tmp_6 = nn.Sequential(*tmp_block(8 * self.ch, 8 * self.ch))

        self.unet_model = UnetGenerator(input_nc=8*self.ch, norm_layer=self.norm_layer_2d, 
                                        ups=self.upsampling, num_downs=nr_layer_g)

 
    def forward(self, input_img):
        # temporal convolutions
        input_img = input_img.unsqueeze(1)
        tmp1 = self.tmp_1(input_img)
        tmp2 = self.tmp_2(tmp1)
        tmp3 = self.tmp_3(tmp2)
        tmp4 = self.tmp_4(tmp3)
        tmp5 = self.tmp_5(tmp4)
        tmp6 = self.tmp_6(tmp5)
        tmp_out = tmp6.squeeze(2)

        # spatial convolutions
        out = self.unet_model(tmp_out)

        return out

