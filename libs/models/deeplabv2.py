#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

# Author: Johnnylu305

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import _ConvBnReLU, _ResLayer, _Stem
from .DRN import drn_d_105

from torchvision import models
from collections import OrderedDict

#from encoding.nn import SyncBatchNorm

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_module("aspp", _ASPP(ch[5], n_classes, atrous_rates))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()


class DeepLabPy_ResNet101(nn.Module, pretrain=False):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self):
        super(DeepLabPy, self).__init__()
        model = models.resnet101(pretrained=pretrain)


        # Layer 3 (OS=16 -> OS=8)
        model.layer3[0].conv2.stride = (1, 1)
        model.layer3[0].downsample[0].stride = (1, 1)
        for m in model.layer3[1:]:
            m.conv2.padding = (2, 2)
            m.conv2.dilation = (2, 2)

        # Layer 4 (OS=32 -> OS=8)
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
        for m in model.layer4[1:]:
            m.conv2.padding = (4, 4)
            m.conv2.dilation = (4, 4)

        # Remove "avgpool" and "fc", and add ASPP
        model = list(model.named_children())[:-2]
        model += [("aspp", _ASPP(2048, 21, [6, 12, 18, 24]))]
        self.model2 = nn.Sequential(OrderedDict(model))

    def freeze_bn(self):
        for n, m in self.named_modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

    def freeze_layer(self):
        for n, m in self.named_parameters():
            break

    def forward(self, x):
        out = self.model2(x)
        return out


if __name__ == "__main__":
    model = DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
