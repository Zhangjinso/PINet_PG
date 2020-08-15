import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F

from .spectral import SpectralNorm

import sys

#refine parsing
#Inception score = (3.2083485, 0.08082621); SSIM score = 0.7749741857614044

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'batch_sync':
        norm_layer = BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='kaiming',
             gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'PInet':
        assert len(input_nc) == 3
        netG = PINetwork(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                           gpu_ids=gpu_ids)
    elif which_model_netG == 'PInet_ca':
        netG = PINetwork(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                           gpu_ids=gpu_ids)
    elif which_model_netG == 'PInet_csa':
        netG = PINetwork(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                           gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='kaiming', gpu_ids=[], use_dropout=False,
             n_downsampling=2):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netD == 'resnet':
        netD = ResnetDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_layers_D,
                                   gpu_ids=[], padding_type='reflect', use_sigmoid=use_sigmoid,
                                   n_downsampling=n_downsampling)
    elif which_model_netD == 'patch':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])

    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



###############Gan loss ############E#############
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


################generator modules###############################

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
class Gated_conv(nn.Module):
    """ Gated convlution Layer"""

    def __init__(self, in_dim, out_dim, kernel_size, stride=3, padding=1, dilation=1, \
        groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(Gated_conv, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.batch_norm = batch_norm
        self.gated_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, stride=stride,  padding=padding, dilation=dilation)
        self.mask_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim , kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.batch_norm2d = nn.BatchNorm2d(out_dim)
        self.sigmoid = nn.Sigmoid()  #


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        res = x
        x = self.gated_conv(x)
        mask = self.mask_conv(res)

        if self.activation is not None:
            x = self.activation(x) * self.sigmoid(mask)
        else:
            x = x*self.sigmoid(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        return x
class Gated_deconv(nn.Module):
    """ Gated deconvlution Layer"""
    """ sale_factor defult as 2"""
    def __init__(self,  in_dim, out_dim, kernel_size, stride=1, padding=1, dilation=1, \
        groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(Gated_deconv, self).__init__()
        self.conv2d = Gated_conv(in_dim, out_dim, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        
    def forward(self, x):
        x = F.upsample(x, scale_factor=2)
        return self.conv2d(x)



##########################generate##########################
class GCNet(torch.nn.Module):
    def __init__(self,  input_nc, out_dim=3, ngf=48, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(GCNet, self).__init__()
        self.gpu_ids = gpu_ids

        self.gcnet = nn.Sequential(
            #input is 5*256*256, but it is full convolution network, so it can be larger than 256
            nn.ReflectionPad2d(3),
            Gated_conv(input_nc, ngf, 7, 1, padding=0),
            # downsample 128
            Gated_conv(ngf, 2*ngf, 3, 2, padding=1),
            Gated_conv(2*ngf, 2*ngf, 3, 1, padding=1),
            Gated_conv(2*ngf, 4*ngf, 3, 2, padding=1),
            #downsample to 64
            Gated_conv(4*ngf, 4*ngf, 3, 1, padding=1),
            Gated_conv(4*ngf, 4*ngf, 3, 1, padding=1),
            Gated_conv(4*ngf, 4*ngf, 3, 1, padding=1),
            # atrous convlution
            Gated_conv(4*ngf, 4*ngf, 3, 1, dilation=2, padding=get_pad(64,3,1,2)),
            Gated_conv(4*ngf, 4*ngf, 3, 1, dilation=4, padding=get_pad(64,3,1,4)),
            Self_Attn(4*ngf, 'relu'),
            Gated_conv(4*ngf, 4*ngf, 3, 1, padding=get_pad(64,3,1)),
            #Self_Attn(4*ngf, 'relu'),
            Gated_conv(4*ngf, 4*ngf, 3, 1, padding=1),
            # upsample
            Gated_deconv(4*ngf, 2*ngf, 3, 1, padding=1),
            #Self_Attn(2*ngf, 'relu'),
            Gated_conv(2*ngf, 2*ngf, 3, 1, padding=1),
            Gated_deconv(2*ngf, ngf, 3, 1, padding=1),

            Gated_conv(ngf, ngf//2, 3, 1, padding=get_pad(256,3,1)),
            #Self_Attn(ngf//2, 'relu'),
            Gated_conv(ngf//2, out_dim, 3, 1, padding=get_pad(128,3,1), activation=None)
            )

        

    def forward(self, input):
        x = self.gcnet(input)
        return x

class PINet(nn.Module):
    def __init__(self,  input_nc, ngf=48, norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[], padding_type='reflect'):
        super(PINet, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]

        cnum=ngf
        shape_in_channel = self.input_nc_s1 + self.input_nc_s2 + 12
        refine_in_channel =  self.input_nc_s1 + self.input_nc_s2 // 2 + 12*2
        self.ParsingNet =  GCNet(shape_in_channel, 12, 64, norm_layer ,  use_dropout, gpu_ids, padding_type)
        self.app_trans_net = nn.Sequential(
            # input is 5*256*256
            nn.ReflectionPad2d(3),
            # input is 5*256*256
            Gated_conv(refine_in_channel, cnum, 7, 1, padding=0),
            # downsample
            Gated_conv(cnum, cnum, 3, 2, padding=1),
            Gated_conv(cnum, 2*cnum, 3, 1, padding=1),
            # downsample
            Gated_conv(2*cnum, 2*cnum, 3, 2, padding=1),
            Gated_conv(2*cnum, 4*cnum, 3, 1, padding=1),
            Gated_conv(4*cnum, 4*cnum, 3, 1, padding=1),
            Gated_conv(4*cnum, 4*cnum, 3, 1, padding=1),
            Gated_conv(4*cnum, 4*cnum, 3, 1, dilation=2, padding=get_pad(64,3,1,2)),
            Gated_conv(4*cnum, 4*cnum, 3, 1, dilation=4, padding=get_pad(64,3,1,4)),
        )
        self.refine_attn = Self_Attn(4*cnum, 'relu')
        self.refine_upsample_net = nn.Sequential(
            Gated_conv(4*cnum, 4*cnum, 3, 1, padding=1),

            Gated_conv(4*cnum, 4*cnum, 3, 1, padding=1),
            Gated_deconv(4*cnum, 2*cnum, 3, 1, padding=1),
            Gated_conv(2*cnum, 2*cnum, 3, 1, padding=1),
            Gated_deconv(2*cnum, cnum, 3, 1, padding=1),

            Gated_conv(cnum, cnum//2, 3, 1, padding=1),
            #Self_Attn(cnum, 'relu'),
            Gated_conv(cnum//2, 3, 3, 1, padding=1, activation=None),
        )


    def forward(self, input):
        x1, x2, x3, x4 = input
        p1, p2 = x2.split(18, dim=1)
        parse_input = torch.cat((x1, x3, x2), 1)
        parse = self.ParsingNet(parse_input)
        parse1 = parse.detach()

        app_input = torch.cat((x1,x3, parse1, p2), 1) 
        x = self.app_trans_net(app_input)
        x= self.refine_attn(x)
        x = self.refine_upsample_net(x)
        x = torch.clamp(x, -1., 1.)

        return x, parse


class PINetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        super(PINetwork, self).__init__()
        #assert type(input_nc) == list and len(input_nc) == 2, 'The AttModule take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = PINet(input_nc, ngf, norm_layer, use_dropout, gpu_ids, padding_type)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

##########################dicrminators#################

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [SpectralNorm(nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetDiscriminator(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
                 padding_type='reflect', use_sigmoid=False, n_downsampling=2):
        assert (n_blocks >= 0)
        super(ResnetDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias)),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # n_downsampling = 2
        if n_downsampling <= 2:
            for i in range(n_downsampling):
                mult = 2 ** i
                model += [SpectralNorm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias)),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
        elif n_downsampling == 3:
            mult = 2 ** 0
            model += [SpectralNorm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias)),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            mult = 2 ** 1
            model += [SpectralNorm(nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias)),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            mult = 2 ** 2
            model += [SpectralNorm(nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                stride=2, padding=1, bias=use_bias)),
                      norm_layer(ngf * mult),
                      nn.ReLU(True)]

        if n_downsampling <= 2:
            mult = 2 ** n_downsampling
        else:
            mult = 4
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        if use_sigmoid:
            model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
class PixelDiscriminator(nn.Module):
    def __init__(self):

       super(PFDiscriminator, self).__init__()


       self.model=nn.Sequential(
           nn.Conv2d(256, 512,kernel_size=4, stride=2,padding=1),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(512, 512,kernel_size=4, stride=2,padding=1),
           nn.InstanceNorm2d(512),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(512, 512,kernel_size=4, stride=2,padding=1)

       )

    def forward(self, input):
        return self.model(input)
