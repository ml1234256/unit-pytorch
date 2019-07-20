import torch
from torch import nn
from torch.autograd import Variable
from utils import weights_init


class ResBlock(nn.Module):
    def __init__(self, channel, padding_type='reflect'):
        super(ResBlock, self).__init__()

        self.conv_block = nn.Sequential(
            Conv2dBlock(channel, channel, padding_type=padding_type, norm='instance', activation='relu'),
            Conv2dBlock(channel, channel, padding_type=padding_type, norm='instance', activation='none')
        )

        #self.conv_block.apply(weights_init)

    def forward(self, x):
        conv_block = self.conv_block(x)
        return conv_block + x


class Conv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding_size=1, padding_type='zero',
                 norm='none', activation='lrelu', conv = 'conv', output_padding=0):
        super(Conv2dBlock, self).__init__()
        model = []
        padding = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(padding_size)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(padding_size)]
        elif padding_type == 'zero':
            padding = padding_size
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if conv == 'conv':
            model += [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        elif conv == 'deconv':
            model += [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=True)]

        if norm == 'bn':
            norm = nn.BatchNorm2d(out_ch)
        elif norm == 'instance':
            norm = nn.InstanceNorm2d(out_ch)
        elif norm == 'none':
            norm = None
        else:
            raise NotImplementedError('normalization [%s] is not implemented' % norm)

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            activation = nn.Tanh()
        elif activation == 'none':
            activation = None
        else:
            raise NotImplementedError('activation [%s] is not implemented' % activation)

        if norm:
            model += [norm]
        if activation:
            model += [activation]

        self.model = nn.Sequential(*model)
        self.model.apply(weights_init)

    def forward(self, x):
        out = self.model(x)
        return out


class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        # if self.training is False:
        #     return x
        if torch.cuda.is_available():
            noise = Variable(torch.randn(x.size()).cuda(x.data.get_device()))
        else:
            noise = Variable(torch.randn(x.size()))
        return x + noise
