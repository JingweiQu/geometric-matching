from __future__ import print_function, division
import torch
import torch.nn as nn

# def deconv(in_channel, out_channel):
#     layer = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1)
    # nn.init.constant_(layer.weight, 1.0)
    # nn.init.constant_(layer.bias, 0)
    # return layer

def deconv(in_channel,
         out_channel,
         kernel_size=3,
         stride=1,
         dilation=1,
         bias=False,
         transposed=False):

    if transposed:
        layer = nn.ConvTranspose2d(in_channel,
                                   out_channel,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=1,
                                   output_padding=1,
                                   dilation=dilation,
                                   bias=bias)
        w = torch.Tensor(kernel_size, kernel_size)
        center = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
        for y in range(kernel_size):
            for x in range(kernel_size):
                w[y, x] = (1 - abs((x - center) / stride)) * (1 - abs((y - center) / stride))
        layer.weight.data.copy_(w.div(in_channel).repeat(in_channel, out_channel, 1, 1))
    if bias:
        nn.init.constant_(layer.bias, 0)
    return layer

def bn(channel):
    layer = nn.BatchNorm2d(channel)
    nn.init.constant_(layer.weight, 1)
    nn.init.constant_(layer.bias, 0)
    return layer

class MaskRegression(nn.Module):
    """
    Do regression to object mask
    """
    def __init__(self):
        super(MaskRegression, self).__init__()
        # Feature map.shape: (batch_size, 1249, 15, 15) to (batch_size, 512, 30, 30) to (batch_size, 256, 60, 60) to
        # (batch_size, 64, 120, 120) to (batch_size, 1, 240, 240)
        block1 = list()
        block2 = list()
        block3 = list()
        block4 = list()

        # block1.append(nn.ConvTranspose2d(1249, 512, kernel_size=3, stride=2, padding=1, output_padding=1))
        block1.append(deconv(1249, 512, stride=2, transposed=True))
        block1.append(bn(512))
        block1.append(nn.ReLU(inplace=True))
        self.block1 = nn.Sequential(*block1)

        # block2.append(nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1))
        block2.append(deconv(512, 256, stride=2, transposed=True))
        block2.append(bn(256))
        block2.append(nn.ReLU(inplace=True))
        self.block2 = nn.Sequential(*block2)

        # block3.append(nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1))
        block3.append(deconv(256, 64, stride=2, transposed=True))
        block3.append(bn(64))
        block3.append(nn.ReLU(inplace=True))
        self.block3 = nn.Sequential(*block3)

        # block4.append(nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1))
        block4.append(deconv(64, 1, stride=2, transposed=True))
        # block4.append(nn.Sigmoid())
        self.block4 = nn.Sequential(*block4)

    def forward(self, x, x_3, x_2, x_1):
        # _, x3, x2, x1 = features
        x = self.block1(x)
        x = self.block2(x + x_3)
        x = self.block3(x + x_2)
        x = self.block4(x + x_1)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)
        return x