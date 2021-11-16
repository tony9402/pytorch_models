import torch
import torch.nn as nn
import torch.nn.functional as F

"""
ResNet
 * paper : https://arxiv.org/pdf/1512.03385.pdf
 * source code : https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html

 - 
"""

def conv3x3(in_planes, out_planes, stride, padding=1, bias = False):
    return nn.Conv2d(in_planes, out_planes, \
        kernel_size = 3,                    \
        stride      = stride,               \
        padding     = padding,              \
        bias        = bias
    )

def conv1x1(in_planes, out_planes, stride, padding=0, bias = False):
    return nn.Conv2d(in_planes, out_planes, \
        kernel_size = 1,                    \
        stride      = stride,               \
        padding     = padding,              \
        bias        = bias
    )

"""

"""
class BasicBlock(nn.Module):
    mul = 1
    def __init__(self, in_planes, out_planes, stride = 1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.conv2 = conv3x3(out_planes, out_planes, 1)

        self.bn1   = nn.BatchNorm2d(out_planes)
        self.bn2   = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, out_planes, stride),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out  = self.conv1(x)
        out  = self.bn1(out)
        out  = F.relu(out)
        out  = self.conv2(out)
        out  = self.bn2(out)
        out += self.shortcut(x)
        out  = F.relu(out)
        return out

"""

"""
class BottleNect(nn.Module):
    mul = 4
    def __init__(self, in_planes, out_planes, stride = 1):
        super(BottleNect, self).__init__()

        self.conv1 = conv1x1(in_planes, out_planes, stride)
        self.conv2 = conv3x3(out_planes, out_planes, 1, 1)
        self.conv3 = conv1x1(out_planes, out_planes * self.mul, 1)

        self.bn1   = nn.BatchNorm2d(out_planes)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.bn3   = nn.BatchNorm2d(out_planes * self.mul)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes * self.mul:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, out_planes * self.mul, stride),
                nn.BatchNorm2d(out_planes * self.mul)
            )

    def forward(self, x):
        out  = self.conv1(x)
        out  = self.bn1(out)
        out  = F.relu(out)
        out  = self.conv2(out)
        out  = self.bn2(out)
        out  = F.relu(out)
        out  = self.conv3(out)
        out  = self.bn3(out)
        out += self.shortcut(x)
        out  = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 1000):
        super(ResNet, self).__init__()

        # 7x7, 64 channels, stride 2 in paper
        self.in_planes = 64 

        # RGB channel -> 64 channels
        self.conv    = nn.Conv2d(3, self.in_planes, kernel_size = 7, stride = 2, padding = 3)
        self.bn      = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        channels, strides = [ 64, 128, 256, 512 ], [ 1, 2, 2, 2 ]

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear  = nn.Linear(512 * block.mul, num_classes)

    def _make_layer(self, block, out_planes, num_block, stride):
        layers  = [ block(self.in_planes, out_planes, stride) ]
        self.in_planes = block.mul * out_planes
        for i in range(num_block - 1):
            layers.append(block(self.in_planes, out_planes, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(BottleNect, [3, 4, 6, 3])

def ResNet101():
    return ResNet(BottleNect, [3, 4, 23, 3])

def ResNet152():
    return ResNet(BottleNect, [3, 8, 36, 3])
