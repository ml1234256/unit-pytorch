# from torch import nn
from ops import *


class UnitDis(nn.Module):
    def __init__(self, opt):
        super(UnitDis, self).__init__()
        padding_type = opt.padding_type
        in_ch = opt.in_ch
        ch = opt.ch
        n_dis_down = opt.n_dis_down
        n_share_dis = opt.n_share_dis

        # def dis(self):
        dis = []
        dis += [Conv2dBlock(in_ch, ch, kernel_size=7, stride=2, padding_size=3, padding_type=padding_type)]
        for i in range(n_dis_down):
            dis += [Conv2dBlock(ch, ch*2, kernel_size=3, stride=2, padding_size=1, padding_type=padding_type)]
            ch *= 2
        dis = nn.Sequential(*dis)
        # return dis, ch

        # def share_dis(self):
        share_dis = []
        for i in range(n_share_dis):
            share_dis += [Conv2dBlock(ch, ch*2, kernel_size=3, stride=2, padding_size=1, padding_type=padding_type)]
            ch *= 2
        share_dis += [Conv2dBlock(ch, 1, kernel_size=1, stride=1, padding_size=0, padding_type='zero')]
        share_dis = nn.Sequential(*share_dis)
        # return share_dis

        self.disA = dis
        self.disB = dis
        self.share_dis = share_dis

    def get_model(self, name):
        if name == 'dis':
            return self.disA
        elif name == 'share':
            return self.share_dis

    # def cuda(self, gpu):
    #     self.disA = self.disAcuda(gpu)
    #     self.disB = self.disB.cuda(gpu)
    #     self.share_dis = self.share_dis(gpu)

    def forward(self, x_a, x_b):
        dis_a = self.share_dis(self.disA(x_a))
        dis_b = self.share_dis(self.disB(x_b))
        # dis_a = dis_a.view(-1)
        # out_a = []
        # out_a.append(dis_a)
        # dis_b.view(-1)
        # out_b = []
        # out_b.append(dis_a)
        return dis_a, dis_b


class UnitGen(nn.Module):
    def __init__(self, opt):
        super(UnitGen, self).__init__()
        self.in_ch = opt.in_ch
        self.ch = opt.ch
        self.padding_type = opt.padding_type
        self.n_enc_down = opt.n_enc_down
        self.n_enc_res = opt.n_enc_res
        self.n_dec_res = opt.n_dec_res
        self.n_share_enc = opt.n_share_enc
        self.n_share_dec = opt.n_share_dec
        self.h_ch = self.ch * pow(2, self.n_enc_down)


    # def encoder(self):
        encoder = []
        encoder += [Conv2dBlock(self.in_ch, self.ch, kernel_size=7, stride=1, padding_size=3,
                                padding_type=self.padding_type)]
        e_ch = self.ch
        for i in range(self.n_enc_down):
            encoder += [Conv2dBlock(e_ch, e_ch*2, kernel_size=3, stride=2, padding_size=1, padding_type=self.padding_type)]
            e_ch *= 2
        for i in range(self.n_enc_res):
            encoder += [ResBlock(e_ch, padding_type=self.padding_type)]
        encoder = nn.Sequential(*encoder)
        # return encoder

    # def share_encoder(self):
        share_encoder = []
        for i in range(self.n_share_enc):
            share_encoder += [ResBlock(self.h_ch, padding_type=self.padding_type)]
        share_encoder += [GaussianNoiseLayer()]
        share_encoder = nn.Sequential(*share_encoder)
        # return share_encoder

    # def share_decoder(self):
        share_decoder = []
        for i in range(self.n_share_dec):
            share_decoder += [ResBlock(self.h_ch, padding_type=self.padding_type)]
        share_decoder = nn.Sequential(*share_decoder)
        # return share_decoder

    # def decoder(self):
        d_ch = self.h_ch
        decoder = []
        for i in range(self.n_dec_res):
            decoder += [ResBlock(d_ch, padding_type=self.padding_type)]
        for i in range(self.n_enc_down):
            decoder += [Conv2dBlock(d_ch, d_ch//2, kernel_size=3, stride=2, padding_size=1, padding_type='zero',
                                    conv='deconv', output_padding=1)]
            d_ch = d_ch//2
        decoder += [Conv2dBlock(d_ch, self.in_ch, kernel_size=1, stride=1, padding_size=0, padding_type='zero',
                                norm='none', activation='tanh')]
        decoder = nn.Sequential(*decoder)
        # return decoder

        self.encoA = encoder
        self.encoB = encoder
        self.share_encoder = share_encoder
        self.share_decoder = share_decoder
        self.decoA = decoder
        self.decoB = decoder


    def get_model(self, name):
        if name == 'encoA':
            return self.encoA
        elif name == 'encoB':
            return self.encoB
    #
    # def cuda(self, gpu):
    #     self.encoA = self.encoA.cuda(gpu)
    #     self.encoB = self.encoB.cuda(gpu)
    #     self.share_encoder = self.share_encoder.cuda(gpu)
    #     self.share_decoder =self.share_decoder.cuda(gpu)
    #     self.decoA = self.decoA(gpu)
    #     self.decoB = self.decoB.cuda(gpu)

    def forward(self, x_a, x_b):
        enc_a = self.encoA(x_a)
        enc_b = self.encoB(x_b)
        shared_z = self.share_encoder(torch.cat((enc_a, enc_b), 0))
        shared_h = self.share_decoder(shared_z)
        dec_a = self.decoA(shared_h)
        dec_b = self.decoB(shared_h)
        x_aa, x_ba = torch.split(dec_a, x_a.size(0), dim=0)
        x_ab, x_bb = torch.split(dec_b, x_a.size(0), dim=0)
        return x_aa, x_ab, x_ba, x_bb, shared_z

    def forwarda2b(self, x_a):
        out = self.encoA(x_a)
        shared = self.share_encoder(out)
        out = self.share_decoder(shared)
        out = self.decoB(out)
        return out, shared

    def forwardb2a(self, x_b):
        out = self.encoB(x_b)
        shared = self.share_encoder(out)
        out = self.share_decoder(shared)
        out = self.decoA(out)
        return out, shared

