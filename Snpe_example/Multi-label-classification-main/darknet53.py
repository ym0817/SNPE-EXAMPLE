
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
from collections import OrderedDict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
# ---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers, class_num=None):
        super(DarkNet, self).__init__()
        self.inplanes = 16
        self.class_num = class_num
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([16, 32], layers[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([32, 64], layers[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([64, 128], layers[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([128, 256], layers[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([256, 512], layers[4])

        # self.layers_out_filters = [64, 128, 256, 512, 1024]
        self.dense = nn.AdaptiveAvgPool2d(1)
        self.layer7 = nn.Conv2d(512, class_num, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(class_num)
        # # 进行权值初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    # ---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    # ---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                            stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out5 = self.layer5(x)

        x = self.dense(out5)
        x = self.layer7(x)
        x = self.bn(x)
        x = x.view(-1, self.class_num)
        x = torch.sigmoid(x)

        return x


def darknet53(pretrained=False, class_num=None, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4], class_num=class_num)
    if pretrained:
        print("darknet use pretrained weight")
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == '__main__':
    input = torch.rand([2, 3, 448, 448])
    net = darknet53(class_num=9)
    a = net(input)
    print(a.shape)
    torch.save(net, "19.pt")