import torch
from torch import nn
from models import *
from utils import *

class UnitLoss():
    def __init__(self, opt, netD, netG):

        self.gpu = opt.gpu
        self.gan_w = opt.gan_w
        self.l1_direct_link_w = opt.l1_direct_link_w
        self.l1_cycle_link_w = opt.l1_cycle_link_w
        self.kl_direct_link_w = opt.kl_direct_link_w
        self.kl_cycle_link_w = opt.kl_cycle_link_w

        self.netD =  netD
        self.netG =  netG


        # if torch.cuda.is_available():
        #     netD = self.netD.cuda(opt.gpu)
        #     netG = self.netG.cuda(opt.gpu)
        # lr = opt.lr
        # self.optim_D = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
        # self.optim_G = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)

        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    # def gen_update(self, realA, realB):
    #     self.netG.zero_grad()
    #     self.gen_loss = self.gen_loss(realA, realB)
    #     self.gen_loss.backward()
    #     self.optim_G.step()
    #
    # def dis_update(self, realA, realB):
    #     self.netD.zero_grad()
    #     self.dis_loss = self.dis_loss(realA, realB)
    #     self.dis_loss.backward()
    #     self.optim_D.step()
    #
    # def get_loss(self):
    #     dis_loss = self.gen_loss
    #     gen_loss = self.dis_loss
    #     return dis_loss, gen_loss

    def _compute_kl(self, mu):
        return torch.mean(torch.pow(mu, 2))

    def gen_loss(self, realA, realB):
        x_aa, x_ab, x_ba, x_bb, shared = self.netG(realA, realB)
        x_aba, shared_aba = self.netG.forwardb2a(x_ab)
        x_bab, shared_bab = self.netG.forwarda2b(x_ba)
        D_fakea, D_fakeb = self.netD(x_ba, x_ab)

        if torch.cuda.is_available():
            real_label = Variable(torch.ones(D_fakea.size())).cuda(self.gpu)
        else:
            real_lable = Variable(torch.ones(D_fakea.size()))

        ad_loss_a = self.bce_loss(self.sigmoid(D_fakea), real_label)
        ad_loss_b = self.bce_loss(self.sigmoid(D_fakeb), real_label)
        l1_loss_a = self.l1_loss(x_aa, realA)
        l1_loss_b = self.l1_loss(x_bb, realB)
        l1_loss_aba = self.l1_loss(x_aba, realA)
        l1_loss_bab = self.l1_loss(x_bab, realB)

        enc_loss = self._compute_kl(shared)
        enc_loss_aba = self._compute_kl(shared_aba)
        enc_loss_bab = self._compute_kl(shared_bab)

        total_loss = self.gan_w * (ad_loss_a + ad_loss_b) + \
            self.l1_direct_link_w * (l1_loss_a + l1_loss_b) + \
            self.l1_cycle_link_w * (l1_loss_aba + l1_loss_bab) + \
            self.kl_direct_link_w * (enc_loss + enc_loss) + \
            self.kl_cycle_link_w * (enc_loss_aba + enc_loss_bab)

        return total_loss
    def dis_loss(self, realA, realB):
        x_aa, x_ab, x_ba, x_bb, shared = self.netG(realA, realB)
        data_a = torch.cat((realA, x_ba), dim=0)
        data_b = torch.cat((realB, x_ab), dim=0)
        out_a, out_b = self.netD(data_a, data_b)

        out_a = self.sigmoid(out_a)
        out_b = self.sigmoid(out_b)

        out_real_a, out_fake_a = torch.split(out_a, out_a.size(0) // 2, dim=0)
        out_real_b, out_fake_b = torch.split(out_b, out_b.size(0) // 2, dim=0)

        if torch.cuda.is_available():
            real_label = Variable(torch.ones(out_real_a.size())).cuda(self.gpu)
            fake_label = Variable(torch.zeros(out_real_a.size())).cuda(self.gpu)
        else:
            real_label = Variable(torch.ones(out_real_a.size()))
            fake_label = Variable(torch.zeros(out_real_a.size()))

        ad_real_loss_a = self.bce_loss(out_real_a, real_label)
        ad_real_loss_b = self.bce_loss(out_real_b, real_label)
        ad_fake_loss_a = self.bce_loss(out_fake_a, fake_label)
        ad_fake_loss_b = self.bce_loss(out_fake_b, fake_label)

        ad_loss_a = ad_real_loss_a + ad_fake_loss_a
        ad_loss_b = ad_real_loss_b + ad_fake_loss_b
        dis_loss = self.gan_w * (ad_loss_a + ad_loss_b)
        return  dis_loss

    # def cuda(self, gpu):
    #     self.netD = self.netD.cuda(gpu)
    #     self.netG = self.netG.cuda(gpu)
