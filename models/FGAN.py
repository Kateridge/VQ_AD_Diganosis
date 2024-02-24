import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class clsBlock(nn.Module):
    def __init__(self, in_channels,out_channels,kernel=3,stride=1,padding=1,do_norm=True,do_relu=True,relufactor=0):
        super(clsBlock,self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride, padding, padding_mode="zeros")
        if do_norm:
            self.norm = nn.InstanceNorm3d(out_channels)
        else:
            self.norm = None
        if do_relu:
            if relufactor == 0:
                self.relu=nn.ReLU(inplace=True)
            else:
                self.relu=nn.LeakyReLU(relufactor,inplace=True)
        else:
            self.relu=None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x=self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Classifier(nn.Module):
    def __init__(self, in_channels=1, features=[16, 32, 64, 64, 64]):
        super(Classifier,self).__init__()
        self.linear=nn.Linear(5120 ,2)
        self.l1_loss = nn.L1Loss()
        self.max_pool=nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding =(1, 1, 1))
        self.average_pool=nn.AvgPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding =(1, 1, 1))
        self.layer_1=clsBlock(in_channels, features[0], 3, 1, 1)
        self.layer_2=clsBlock(features[0], features[1], 3, 1, 1)
        self.layer_3=clsBlock(features[1], features[2], 3, 1, 1)
        self.layer_4=clsBlock(features[2], features[3], 3, 1, 1)
        self.layer_5=clsBlock(features[3], features[4], 3, 1, 1)

    def forward(self, x, PET = None):
        loss = 0
        layer_1=self.layer_1(x)
        layer_1_0=self.max_pool(layer_1)
        layer_2=self.layer_2(layer_1_0)
        layer_2_0=self.max_pool(layer_2)
        layer_3=self.layer_3(layer_2_0)
        layer_3_0=self.max_pool(layer_3)
        layer_4=self.layer_4(layer_3_0)
        layer_4_0=self.max_pool(layer_4)
        layer_5=self.layer_5(layer_4_0)
        layer_5_0=self.average_pool(layer_5)
        # out = layer_5_0.reshape(1,-1)
        out = rearrange(layer_5_0, 'b c h w d -> b (c h w d)')
        # pred = self.linear(out)

        # if PET != None:
        #     layer_1_true= self.layer_1(PET)
        #     layer_1_0_true= self.max_pool(layer_1_true)
        #     layer_2_true= self.layer_2(layer_1_0_true)
        #     layer_2_0_true= self.max_pool(layer_2_true)
        #     layer_3_true= self.layer_3(layer_2_0_true)
        #     layer_3_0_true= self.max_pool(layer_3_true)
        #     layer_4_true= self.layer_4(layer_3_0_true)
        #     layer_4_0_true= self.max_pool(layer_4_true)
        #     layer_5_true= self.layer_5(layer_4_0_true)
        #     loss = self.l1_loss(layer_1,layer_1_true)+self.l1_loss(layer_2,layer_2_true)+self.l1_loss(layer_3,layer_3_true)+self.l1_loss(layer_4,layer_4_true)+self.l1_loss(layer_5,layer_5_true)
        #     return loss
        # else:
        return out

class Block(nn.Module):
    def __init__(self, in_channels,out_channels,kernel=3,stride=1,padding=1,mode="zeros",do_norm=True,do_relu=True,relufactor=0,norm = "Instancenorm"):
        super(Block,self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride, padding, padding_mode=mode)
        if do_norm:
            if norm == "Instancenorm":
                self.norm = nn.InstanceNorm3d(out_channels)
            else:
                self.norm = nn.BatchNorm3d(out_channels)
        else:
            self.norm = None
        if do_relu:
            if relufactor == 0:
                self.relu=nn.ReLU(inplace=True)
            else:
                self.relu=nn.LeakyReLU(relufactor,inplace=True)
        else:
            self.relu=None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x=self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Deconv_Block(nn.Module):
    def __init__(self, in_channels,out_channels,kernel=3,stride=1,padding=1,mode="zeros",do_norm=True,do_relu=True,relufactor=0,norm = "Instancenorm"):
        super(Deconv_Block,self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride, padding, padding_mode=mode)
        if do_norm:
            if norm == "Instancenorm":
                self.norm = nn.InstanceNorm3d(out_channels)
            else:
                self.norm = nn.BatchNorm3d(out_channels)
        else:
            self.norm = None
        if do_relu:
            if relufactor == 0:
                self.relu=nn.ReLU(inplace=True)
            else:
                self.relu=nn.LeakyReLU(relufactor,inplace=True)
        else:
            self.relu=None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x=self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class resnet_block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(resnet_block,self).__init__()
        self.relu=nn.ReLU(inplace=True)
        self.layer_1=Block(in_channels, out_channels, 3, 1, 1)
        self.layer_2=Block(out_channels, out_channels, 3, 1, 1, do_relu=False)

    def forward(self, x):
        layer_1=self.layer_1(x)
        layer_2=self.layer_2(layer_1)
        return self.relu(layer_2 + x)

class Generator(nn.Module):
    def __init__(self, input_nc = 1, output_nc = 1, ngf= 16, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=3, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm3d

        model = [nn.ReflectionPad3d(3),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad3d(3)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)  # add skip connections
        return out

class Discriminator(nn.Module):
    def __init__(self, input_nc = 1, ndf= 8, n_layers=4, norm_layer=nn.BatchNorm3d):
        super(Discriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

if __name__ == "__main__":
    x = torch.randn((1, 1, 128, 144, 128))
    # model = Discriminator()
    # model = Generator()
    model = Classifier()
    out= model(x)
    print(out.shape)
    print(out)