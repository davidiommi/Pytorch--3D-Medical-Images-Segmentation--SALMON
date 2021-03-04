from train import *
from torch.nn import init
import monai
from torch.optim import lr_scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.epochs/2) / float(opt.epochs/2 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    # print('learning rate = %.7f' % lr)


from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d
from torch.nn import ReLU, Sigmoid
import torch


def build_net():
    from init import Options
    opt = Options().parse()
    from monai.networks.layers import Norm
    from monai.networks.layers.factories import split_args
    act_type, args = split_args("RELU")

    # create Unet
    Unet = monai.networks.nets.UNet(
        dimensions=3,
        in_channels=opt.in_channels,
        out_channels=opt.out_channels,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        act=act_type,
        num_res_units=3,
        dropout=0.2,
        norm=Norm.BATCH,

    )

    class UNet_David(Module):
        # __                            __
        #  1|__   ________________   __|1
        #     2|__  ____________  __|2
        #        3|__  ______  __|3
        #           4|__ __ __|4
        # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

        def __init__(self, feat_channels=[32, 64, 128, 256, 512], residual='conv'):
            # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

            super(UNet_David, self).__init__()

            class Conv3D_Block(Module):

                def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

                    super(Conv3D_Block, self).__init__()

                    self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel,
                               stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

                    self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel,
                               stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

                    self.residual = residual

                    if self.residual is not None:
                        self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

                def forward(self, x):

                    res = x

                    if not self.residual:
                        return self.conv2(self.conv1(x))
                    else:
                        return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

            class Deconv3D_Block(Module):

                def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
                    super(Deconv3D_Block, self).__init__()

                    self.deconv = Sequential(
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                                        stride=(stride, stride, stride), padding=(padding, padding, padding),
                                        output_padding=1, bias=True),
                        ReLU())

                def forward(self, x):
                    return self.deconv(x)

            class ChannelPool3d(AvgPool1d):

                def __init__(self, kernel_size, stride, padding):
                    super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
                    self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

                def forward(self, inp):
                    n, c, d, w, h = inp.size()
                    inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
                    pooled = self.pool_1d(inp)
                    c = int(c / self.kernel_size[0])
                    return inp.view(n, c, d, w, h)

            # Encoder downsamplers
            self.pool1 = MaxPool3d((2, 2, 2))
            self.pool2 = MaxPool3d((2, 2, 2))
            self.pool3 = MaxPool3d((2, 2, 2))
            self.pool4 = MaxPool3d((2, 2, 2))

            # Encoder convolutions
            self.conv_blk1 = Conv3D_Block(opt.in_channels, feat_channels[0], residual=residual)
            self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
            self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
            self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
            self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

            # Decoder convolutions
            self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
            self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
            self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
            self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

            # Decoder upsamplers
            self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
            self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
            self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
            self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

            # Final 1*1 Conv Segmentation map
            self.one_conv = Conv3d(feat_channels[0], opt.out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def forward(self, x):
            # Encoder part

            x1 = self.conv_blk1(x)

            x_low1 = self.pool1(x1)
            x2 = self.conv_blk2(x_low1)

            x_low2 = self.pool2(x2)
            x3 = self.conv_blk3(x_low2)

            x_low3 = self.pool3(x3)
            x4 = self.conv_blk4(x_low3)

            x_low4 = self.pool4(x4)
            base = self.conv_blk5(x_low4)

            # Decoder part

            d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
            d_high4 = self.dec_conv_blk4(d4)

            d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
            d_high3 = self.dec_conv_blk3(d3)
            d_high3 = Dropout3d(p=0.5)(d_high3)

            d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
            d_high2 = self.dec_conv_blk2(d2)
            d_high2 = Dropout3d(p=0.5)(d_high2)

            d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
            d_high1 = self.dec_conv_blk1(d1)

            seg = self.one_conv(d_high1)

            return seg

    # create HighResNet
    HighResNet = monai.networks.nets.HighResNet(
        spatial_dims=3,
        in_channels=opt.in_channels,
        out_channels=opt.out_channels,
    )

    if opt.net == 'Unet_Monai':
        network = Unet
    elif opt.net == 'HighResNet':
        network = HighResNet
    elif opt.net == 'Unet_David':
        network = UNet_David(residual='pool')
    else:
        raise NotImplementedError

    init_weights(network, init_type='normal')

    return network


if __name__ == '__main__':
    import time
    import torch
    from torch.autograd import Variable
    from torchsummaryX import summary
    from torch.nn import init

    opt = Options().parse()

    torch.cuda.set_device(0)
    network = build_net()
    net = network.cuda().eval()

    print(net)

    data = Variable(torch.randn(int(opt.batch_size), int(opt.in_channels), int(opt.patch_size[0]), int(opt.patch_size[1]), int(opt.patch_size[2]))).cuda()

    out = net(data)

    summary(net,data)
    print("out size: {}".format(out.size()))






