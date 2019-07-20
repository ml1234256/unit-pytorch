import torch
import argparse
from train import train
from test import test
from models import UnitGen
from utils import load_net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='train', help='train or test')
    parser.add_argument('--dataroot', default='./data/gopro_unpair/train_unpair', help='path to dataset')
    parser.add_argument('--out_dir', default='./out', help='output direction')
    parser.add_argument('--epoch', type=int, default=300, help='the starting epoch count')
    parser.add_argument('--checkpoints_dir', default='./checkpoints', help='The direction model saved')
    parser.add_argument('--load_epoch', type=int, default=1, help='load epoch checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--in_ch', type=int, default=3, help='input channel ')
    parser.add_argument('--ch', type=int, default=64, help=' ')
    parser.add_argument('--padding_type', default='reflect', help=' ')
    parser.add_argument('--fineSize', default=256, help='')
    parser.add_argument('--loadSizeX', default=360, help='')
    parser.add_argument('--loadSizeY', default=360, help='')
    parser.add_argument('--niter', type=int, default=150, help='of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=150, help='of iter to linearly decay learning rate to zero')

    parser.add_argument('--n_enc_down', type=int, default=2, help=' ')
    parser.add_argument('--n_enc_res', type=int, default=3, help=' ')
    parser.add_argument('--n_dec_res', type=int, default=3, help=' ')
    parser.add_argument('--n_share_enc', type=int, default=1, help=' ')
    parser.add_argument('--n_share_dec', type=int, default=1, help=' ')
    parser.add_argument('--n_dis_down', type=int, default=1, help=' ')
    parser.add_argument('--n_share_dis', type=int, default=4, help=' ')
    parser.add_argument('--gan_w', type=int, default=4, help=' ')
    parser.add_argument('--l1_direct_link_w', type=int, default=4, help=' ')
    parser.add_argument('--l1_cycle_link_w', type=int, default=4, help=' ')
    parser.add_argument('--kl_direct_link_w', type=int, default=4, help=' ')
    parser.add_argument('--kl_cycle_link_w', type=int, default=4, help=' ')
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end '
                                                                       'of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    return parser.parse_args()


opt = parse_args()
if opt.model == 'train':
    train(opt)

if opt.model == 'test':
    netG = UnitGen(opt)
    load_net(netG, opt.checkpoints_dir, 'G', opt.load_epoch )
    if torch.cuda.is_available():
        netG = netG.cuda()
    
    test(opt, netG)
