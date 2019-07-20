from torch.utils.data import DataLoader
from torch.autograd import Variable
from losses import UnitLoss
from models import UnitDis, UnitGen
from utils import *

def train(opt):
    traindata = ReadConcat(opt)
    trainset = DataLoader(traindata, batch_size=opt.batchSize, shuffle=True)

    netD = UnitDis(opt)
    netG = UnitGen(opt)
    if torch.cuda.is_available():
        netD = netD.cuda(opt.gpu)
        netG = netG.cuda(opt.gpu)

    lr = opt.lr
    optim_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    optim_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)

    unit_loss = UnitLoss(opt, netD, netG)

    for e in range(opt.epoch):
        for i, data in enumerate(trainset):
            data_a = data['A']
            data_b = data['B']

            if torch.cuda.is_available():
                data_a = data_a.cuda(opt.gpu)
                data_b = data_b.cuda(opt.gpu)

            data_a = Variable(data_a)
            data_b = Variable(data_b)

            for iter_d in range(1):
                # trainer.dis_update(data_a, data_b)
                netD.zero_grad()
                dis_loss = unit_loss.dis_loss(data_a, data_b)
                dis_loss.backward()
                optim_D.step()
            netG.zero_grad()
            gen_loss = unit_loss.gen_loss(data_a, data_b)
            gen_loss.backward()
            optim_G.step()

            if i % 50 == 0:

                print('{}/{}: lossD:{}, lossG:{}'.format(i, e, dis_loss, gen_loss))

        if e > opt.niter:
            update_lr(optim_D, opt.lr, opt.niter_decay)
            update_lr(optim_G, opt.lr, opt.niter_decay)

        if e % opt.save_epoch_freq == 0:
            save_net(netG, opt.checkpoints_dir, 'G', e)
            save_net(netD, opt.checkpoints_dir, 'D', e)
