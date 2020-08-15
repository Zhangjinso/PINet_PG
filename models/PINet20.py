import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from . import network20 as networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss

import sys
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'PInet':
        assert opt.dataset_mode == 'keypoint'
        model = TransferModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

class TransferModel(nn.Module):
    def __init__(self):
        super(TransferModel, self).__init__()

    def name(self):
        return 'TransferModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)


        nb = opt.batchSize
        size = opt.fineSize
        self.input_P1_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_KP1_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_P2_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_KP2_set = self.Tensor(nb, opt.BP_input_nc, size, size)

        self.input_SPL1_set = self.Tensor(nb, 1,size, size)
        self.input_SPL2_set = self.Tensor(nb, 1,size, size)
        self.input_SPL1_onehot_set = self.Tensor(nb, 12, size, size)
        self.input_SPL2_onehot_set = self.Tensor(nb, 12, size, size)

        self.input_syn_set = self.Tensor(nb, opt.P_input_nc, size, size)

        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc, opt.P_input_nc]
        self.netG = networks.define_G(input_nc, 
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(3+18, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, 'instance', use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, 'instance', use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            #define shape loss
            if False: #self._opt.mask_bce:
                self.parseLoss = torch.nn.BCELoss()
            else:
                self.parseLoss = CrossEntropyLoss2d()

            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            else:
                raise Excption('Unsurportted type of L1!')
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_P1, input_KP1, input_SPL1 = input['P1'], input['KP1'], input['SPL1']
        input_P2, input_KP2, input_SPL2 = input['P2'], input['KP2'], input['SPL2']

 
        input_SPL1_onehot = input['SPL1_onehot']
        input_SPL2_onehot = input['SPL2_onehot']
        self.input_SPL1_onehot_set.resize_(input_SPL1_onehot.size()).copy_(input_SPL1_onehot)
        self.input_SPL2_onehot_set.resize_(input_SPL2_onehot.size()).copy_(input_SPL2_onehot)

        self.input_SPL1_set.resize_(input_SPL1.size()).copy_(input_SPL1)
        self.input_SPL2_set.resize_(input_SPL2.size()).copy_(input_SPL2)

        
        #qinput_syn = input_syn[:,:,:,40:216]


        self.input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        self.input_KP1_set.resize_(input_KP1.size()).copy_(input_KP1)
        self.input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        self.input_KP2_set.resize_(input_KP2.size()).copy_(input_KP2)

        self.image_paths = input['P1_path'][0] + '___' + input['P2_path'][0]


    def forward(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_KP1 = Variable(self.input_KP1_set)
        self.input_SPL1 = Variable(self.input_SPL1_set)


        self.input_P2 = Variable(self.input_P2_set)
        self.input_KP2 = Variable(self.input_KP2_set)
        self.input_SPL2 = Variable(self.input_SPL2_set) #bs 1 256 176
#        print(self.input_SPL2.shape)
        self.input_SPL1_onehot = Variable(self.input_SPL1_onehot_set)
        self.input_SPL2_onehot = Variable(self.input_SPL2_onehot_set)

        
        G_input = [self.input_P1,
                   torch.cat((self.input_KP1, self.input_KP2), 1), 
                  self.input_SPL1_onehot, self.input_SPL2_onehot]
        self.fake_p2, self.fake_parse = self.netG(G_input)


    def test(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_KP1 = Variable(self.input_KP1_set)
        self.input_SPL1 = Variable(self.input_SPL1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_KP2 = Variable(self.input_KP2_set)
        self.input_SPL2 = Variable(self.input_SPL2_set)

        self.input_SPL1_onehot = Variable(self.input_SPL1_onehot_set)
        self.input_SPL2_onehot = Variable(self.input_SPL2_onehot_set)

        G_input = [self.input_P1,
                   torch.cat((self.input_KP1, self.input_KP2), 1), 
                   self.input_SPL1_onehot, self.input_SPL2_onehot]
        self.fake_p2, self.fake_parse = self.netG(G_input)


    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.pred_fake = self.fake_PB_pool.query(torch.cat((self.input_KP2, self.fake_p2 ),1).data)
        self.pred_real = torch.cat((self.input_KP2, self.input_P2),1)
        self.loss_DPB_fake = self.backward_D_basic(self.netD_PB ,self.pred_real, self.pred_fake).item()

        self.pred_fake = self.fake_PP_pool.query(torch.cat((self.fake_p2,self.input_P1),1).data)
        self.pred_real = torch.cat((self.input_P2,self.input_P1),1)
        self.loss_DPP_fake = self.backward_D_basic(self.netD_PP ,self.pred_real, self.pred_fake).item()


    def backward_G(self):
       
        mask = self.input_SPL2.squeeze(1).long()
        self.maskloss1 = self.parseLoss(self.fake_parse, mask)

        L1_per = self.criterionL1(self.fake_p2, self.input_P2)
        self.loss_G_L1 = L1_per[0]
        pred_fake = self.netD_PB(torch.cat((self.input_KP2, self.fake_p2),1))
        pred_fake_pp = self.netD_PP(torch.cat((self.fake_p2,self.input_P1),1))

        self.L1 = L1_per[1]
        self.per = L1_per[2]
        self.loss_G_GAN = (self.criterionGAN(pred_fake, True) + self.criterionGAN(pred_fake_pp, True))/2

        self.loss_mask =  self.loss_G_L1 + self.loss_G_GAN * self.opt.lambda_GAN+ self.maskloss1
        self.loss_mask.backward()


    def optimize_parameters(self):
        self.forward()
        self.optimizer_D_PB.zero_grad()
        self.optimizer_D_PP.zero_grad()
        self.backward_D()
        self.optimizer_D_PB.step()
        self.optimizer_D_PP.step()
       
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()



    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.opt.with_D_PB or self.opt.with_D_PP:
            ret_errors['L1_plus_perceptualLoss'] = self.loss_G_L1
            ret_errors['percetual'] = self.per
            ret_errors['L1'] = self.L1
            ret_errors['PB'] = self.loss_DPB_fake
            ret_errors['PP'] = self.loss_DPP_fake
            ret_errors['pair_GANloss'] = self.loss_G_GAN.data.item()
            ret_errors['parsing1'] = self.maskloss1.data.item()



        return ret_errors

    def get_current_visuals(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)
        
        input_SPL1 = util.tensor2im(torch.argmax(self.input_SPL1_onehot, axis=1, keepdim=True).data, True)
        input_SPL2 = util.tensor2im(torch.argmax(self.input_SPL2_onehot, axis=1, keepdim=True).data, True)

        input_KP1 = util.draw_pose_from_map(self.input_KP1.data)[0]
        input_KP2 = util.draw_pose_from_map(self.input_KP2.data)[0]

        fake_shape2 = util.tensor2im(torch.argmax(self.fake_parse, axis=1,keepdim=True).data, True)
        fake_p2 = util.tensor2im(self.fake_p2.data)

        vis = np.zeros((height, width*8, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_KP1
        vis[:, width*2:width*3, :] = input_SPL1
        if input_P2.shape[1]==256:
            vis[:, width*3:width*4, :] = input_P2[:,40:216,:]
        else:

            vis[:, width*3:width*4, :] = input_P2
        vis[:, width*4:width*5, :] = input_KP2
        vis[:, width*5:width*6, :] = input_SPL2
        vis[:, width*6:width*7, :] = fake_shape2
        vis[:, width*7:, :] = fake_p2

        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)

        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)

        # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)




###### crossentropy for parsing ########
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, inputs, targets):
        return self.nll_loss(self.softmax(inputs), targets)
