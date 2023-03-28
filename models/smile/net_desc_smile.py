from collections import OrderedDict
from CCNet.cc_attention.functions import CrissCrossAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """
    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x


class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, stride=2)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        x = self.up(x)

        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        x_skip = F.pad(x_skip, (0 ,W_diff ,0,H_diff), mode='reflect')

        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Transfer Learning ResNet as Encoder part of UNet

class SMILE_Net(nn.Module):

    def __init__(self, input_ch=3, nr_types=None, pretrained=True, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.resnet = models.resnet34(pretrained=pretrained)
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
                'Unknown mode `%s` for HoVerNet %s. Only support `original` or `fast`.' % mode
        self.cca = CrissCrossAttention(512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
            # InPlaceABNSync(out_channels),
            nn.BatchNorm2d(512),
            nn.Dropout2d(0.1),
        )

        module_list = [
            # ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            # ("/", self.resnet.conv1) ,
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=3, bias=False)),
            ("bn", self.resnet.bn1),
            ("relu", nn.ReLU(inplace=True)),
        ]


        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.con0_maxpool = self.resnet.maxpool
        # up conv

        l = [64, 64, 128, 256, 512]

        self.d0 = self.resnet.layer1
        self.d1 = self.resnet.layer2
        self.d2 = self.resnet.layer3
        self.d3 = self.resnet.layer4

        # final conv
        self.ce = nn.ConvTranspose2d(l[0], self.output_ch, 2, stride=2)
        def create_encoder_branch(input_chanel=input_ch):
            module_list = [

                ("/", nn.Conv2d(input_chanel, 64, 7, stride=1, padding=3, bias=False)),
                ("bn", models.resnet34(pretrained=pretrained).bn1),
                ("relu", nn.ReLU(inplace=True)),
            ]
            conv0 = nn.Sequential(OrderedDict(module_list))
            con0_maxpool = self.resnet.maxpool
            # up conv


            d0 = models.resnet34(pretrained=pretrained).layer1
            d1 = models.resnet34(pretrained=pretrained).layer2
            d2 = models.resnet34(pretrained=pretrained).layer3
            d3 = models.resnet34(pretrained=pretrained).layer4

            # final conv
            encoder = nn.Sequential(
                OrderedDict([("conv0", conv0), ("con0_maxpool", con0_maxpool), ("d0", d0), ("d1", d1), ("d2", d2),("d3", d3) ])

            )
            return encoder
        if nr_types is None:
            self.encoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_encoder_branch(input_chanel=3)),
                        ("hv", create_encoder_branch(input_chanel=3)),
                    ]
                )
            )
        else:
            self.encoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_encoder_branch(input_chanel=3)),
                        ("np", create_encoder_branch(input_chanel=3)),
                        ("hv", create_encoder_branch(input_chanel=3)),
                    ]
                )
            )


        def create_decoder_branch(out_ch=2 ): # ksize=5
            l = [64, 64, 128, 256, 512]

            u3 = ConvUpBlock(l[4], l[3], dropout_rate=0.1)

            u2 = ConvUpBlock(l[3], l[2], dropout_rate=0.1)

            u1 = ConvUpBlock(l[2], l[1], dropout_rate=0.1)

            u0 = ConvUpBlock(l[1], l[0], dropout_rate=0.1)

            module_list = [
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            ce = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0), ("ce", ce),])
            )
            return decoder

        # ksize = 5 if mode == 'original' else 3
        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(out_ch=2)),
                        ("hv", create_decoder_branch(out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(out_ch=nr_types)),
                        ("np", create_decoder_branch(out_ch=2)),
                        ("hv", create_decoder_branch(out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()
        # TODO: pytorch still require the channel eventhough its ignored
        # self.weights_init()

    def forward(self, imgs):
        # print("Use CCA")
        imgs = imgs / 255.0  # to 0-1 range to match XY

        encoder_out_dict = OrderedDict()
        if self.training:
            with torch.set_grad_enabled(not self.freeze):
                for branch_name, branch_desc in self.encoder.items():
                    conv0 = branch_desc[0](imgs)
                    con0_maxpool = branch_desc[1](conv0)
                    d0 = branch_desc[2](con0_maxpool)
                    d1 = branch_desc[3](d0)
                    d2 = branch_desc[4](d1)
                    d3 = branch_desc[5](d2)

                    cca1 = self.cca(d3)
                    cca2 = self.cca(cca1)
                    d3 = self.bottleneck(torch.cat([d3, cca2], 1))

                    encoder_out_dict[branch_name] = [conv0, d0, d1, d2, d3]
        else:
            for branch_name, branch_desc in self.encoder.items():
                conv0 = branch_desc[0](imgs)
                con0_maxpool = branch_desc[1](conv0)
                d0 = branch_desc[2](con0_maxpool)
                d1 = branch_desc[3](d0)
                d2 = branch_desc[4](d1)
                d3 = branch_desc[5](d2)

                cca1 = self.cca(d3)
                cca2 = self.cca(cca1)

                d3 = self.bottleneck(torch.cat([d3, cca2], 1))

                encoder_out_dict[branch_name] = [conv0, d0, d1, d2, d3]


        out_dict = OrderedDict()
        for out_branch_name, out_branch_desc in self.decoder.items():

            if out_branch_name:

                if out_branch_name=="tp":
                    d = encoder_out_dict["tp"]
                    for branch_name, branch_desc in self.decoder.items():
                        if branch_name=="tp":

                            u3 = branch_desc[0](d[-1], d[-2])
                            u2 = branch_desc[1](u3, d[-3])
                            u1 = branch_desc[2](u2, d[-4])
                            u0 = branch_desc[3](u1, conv0)

                            ce = branch_desc[4](u0)

                            out_dict[branch_name] = ce
                elif out_branch_name == "np":
                    d = encoder_out_dict["np"]
                    for branch_name, branch_desc in self.decoder.items():
                        if branch_name == "np":
                            u3 = branch_desc[0](d[-1], d[-2])
                            u2 = branch_desc[1](u3, d[-3])
                            u1 = branch_desc[2](u2, d[-4])
                            u0 = branch_desc[3](u1, conv0)

                            ce = branch_desc[4](u0)

                            out_dict[branch_name] = ce
                elif out_branch_name == "hv":
                    d = encoder_out_dict["hv"]
                    for branch_name, branch_desc in self.decoder.items():
                        if branch_name == "hv":
                            u3 = branch_desc[0](d[-1], d[-2])
                            u2 = branch_desc[1](u3, d[-3])
                            u1 = branch_desc[2](u2, d[-4])
                            u0 = branch_desc[3](u1, conv0)
                            ce = branch_desc[4](u0)
                            out_dict[branch_name] = ce

        return out_dict


# plain UNet
class UNet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # down conv
        self.c1 = ConvDownBlock(in_c, 16)
        self.c2 = ConvDownBlock(16, 32)
        self.c3 = ConvDownBlock(32, 64)
        self.c4 = ConvDownBlock(64, 128)
        self.cu = ConvDownBlock(128, 256)
        # up conv
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        # final conv
        self.ce = nn.Conv2d(16, out_c, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        return x

####
def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return SMILE_Net(pretrained=True, mode=mode, **kwargs)