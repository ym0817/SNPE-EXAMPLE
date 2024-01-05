# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 2022-02-26
import os
import warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


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

class Conv_BN_Relu(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, strid=1):
        super(Conv_BN_Relu, self).__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=strid, padding=pad)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(0.1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x

class DarkNet(nn.Module):
    fileter = [32, 64, 64, 64, 128, 128]

    def __init__(self, class_num):
        super(DarkNet, self).__init__()
        self.class_num = class_num
        self.layer1 = nn.Sequential(
            Conv_BN_Relu(3, self.fileter[0]),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            Conv_BN_Relu(self.fileter[0], self.fileter[0], strid=2)
        )
        self.layer2 = nn.Sequential(
            Conv_BN_Relu(self.fileter[0], self.fileter[1]),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            Conv_BN_Relu(self.fileter[1], self.fileter[1], strid=2),
        )
        self.layer3 = nn.Sequential(
            Conv_BN_Relu(self.fileter[1], self.fileter[2]),
            Conv_BN_Relu(self.fileter[2], self.fileter[1], kernel_size=1),
            Conv_BN_Relu(self.fileter[1], self.fileter[2]),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            Conv_BN_Relu(self.fileter[2], self.fileter[2], strid=2),
        )
        self.layer4 = nn.Sequential(
            Conv_BN_Relu(self.fileter[2], self.fileter[3]),
            Conv_BN_Relu(self.fileter[3], self.fileter[2], kernel_size=1),
            Conv_BN_Relu(self.fileter[2], self.fileter[3]),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            Conv_BN_Relu(self.fileter[3], self.fileter[3], strid=2),
        )
        self.layer5 = nn.Sequential(
            # SELayer(self.fileter[3]),
            Conv_BN_Relu(self.fileter[3], self.fileter[4]),
            Conv_BN_Relu(self.fileter[4], self.fileter[3], kernel_size=1),
            Conv_BN_Relu(self.fileter[3], self.fileter[4]),
            Conv_BN_Relu(self.fileter[4], self.fileter[3], kernel_size=1),
            Conv_BN_Relu(self.fileter[3], self.fileter[4]),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_BN_Relu(self.fileter[4], self.fileter[4], strid=2),

        )
        self.layer6 = nn.Sequential(
            # SELayer(self.fileter[4]),
            Conv_BN_Relu(self.fileter[5], self.fileter[5]),
            Conv_BN_Relu(self.fileter[5], self.fileter[4], kernel_size=1),
            Conv_BN_Relu(self.fileter[4], self.fileter[5]),
            Conv_BN_Relu(self.fileter[5], self.fileter[4], kernel_size=1),
            Conv_BN_Relu(self.fileter[4], self.fileter[5]),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_BN_Relu(self.fileter[5], self.fileter[5], strid=2),
        )
        # self.dense = nn.AvgPool2d(4) error when onnx model to trt !!  cause wide and height is 1
        # self.dense = nn.Conv2d(self.fileter[5], self.fileter[5], kernel_size=4, stride=4, padding=0)
        self.layer7 = nn.Conv2d(self.fileter[5], class_num, kernel_size=4, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(class_num)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        # if self.train():
        #     x = torch.dropout(x, 0.2, train=True)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
        # x = self.dense(x)
        # x = F.avg_pool2d(x, kernel_size=4)
        x = self.layer7(x)
        x = self.bn(x)
        x = x.view(-1, self.class_num)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    input = torch.rand([1, 3, 448, 448])
    net = DarkNet(9)
    net.eval()
    a = net(input)
    print(a.shape)
    torch.save(net, "19.pt")
