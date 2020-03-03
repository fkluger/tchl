from torchvision.models import resnet as R
from torch import nn, stack
import torch
import torch.nn.functional as F
from torch.utils import model_zoo
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(R.BasicBlock):
    fov_increase = 0
    block_type = '2d'
    block_name = "BasicBlock"

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__(inplanes, planes, stride, downsample)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_3_3(nn.Module):
    expansion = 1
    fov_increase = 2
    block_type = '3d'
    block_name = "BasicBlock_3_3"

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_3_3, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=(1, stride, stride),
                     padding=(1, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=(1, 1, 1),
                               padding=(1, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_3_5(nn.Module):
    expansion = 1
    fov_increase = 3
    block_type = '3d'
    block_name = "BasicBlock_3_5"

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_3_5, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=(1, stride, stride),
                     padding=(1, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(5, 3, 3), stride=(1, 1, 1),
                               padding=(2, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_5_5(nn.Module):
    expansion = 1
    fov_increase = 4
    block_type = '3d'
    block_name = "BasicBlock_5_5"

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_5_5, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(5, 3, 3), stride=(1, stride, stride),
                     padding=(2, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(5, 3, 3), stride=(1, 1, 1),
                               padding=(2, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_5_7(nn.Module):
    expansion = 1
    fov_increase = 5
    block_type = '3d'
    block_name = "BasicBlock_5_7"

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_5_7, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(5, 3, 3), stride=(1, stride, stride),
                     padding=(2, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(7, 3, 3), stride=(1, 1, 1),
                               padding=(3, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_1_3(nn.Module):
    expansion = 1
    fov_increase = 1
    block_type = '3d'
    block_name = "BasicBlock_1_3"

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_1_3, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                     padding=(0, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=(1, 1, 1),
                               padding=(1, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock_1_1(nn.Module):
    expansion = 1
    fov_increase = 0
    block_type = '3d'
    block_name = "BasicBlock_1_1"

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_1_1, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                     padding=(0, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, 1, 1),
                               padding=(0, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):

    def __init__(self, blocks, layers, order='BCDHW'):
        self.block_types = []
        self.order = order
        self.fov_increase = 0
        self.inplanes = 64
        super(ResNet3d, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        planes = self.inplanes
        self.layer1 = self._make_layer(blocks[0], 64, layers[0])
        self.layer2 = self._make_layer(blocks[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(blocks[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(blocks[3], 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 1000)
        self.fc_ = nn.Linear(512, 512)

        self.fc_o = nn.Linear(512, 1)
        self.fc_a = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block.block_type == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=(1, stride, stride), bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            elif block.block_type == '2d':
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=(stride, stride), bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else: assert False

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        self.fov_increase += block.fov_increase
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            self.fov_increase += block.fov_increase

        self.block_types += [block.block_type]

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.order == 'BCDHW':
            x.transpose_(1, 2)
        elif self.order != 'BDCHW':
            assert False, "unknown dim order"

        B = x.shape[0]
        S = x.shape[1]
        D = S

        y = x.reshape([B * S, x.shape[2], x.shape[3], x.shape[4]])

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        C = y.shape[1]
        H = y.shape[2]
        W = y.shape[3]

        y = self.avgpool(y)

        x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])

        x = self.fc_(x)
        x = nn.functional.relu(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        return offset, angle


class ResNet_2_2d_2_3d(nn.Module):

    def __init__(self, blocks, layers, order='BCDHW'):
        self.block_types = []
        self.order = order
        self.fov_increase = 0
        self.inplanes = 64
        super(ResNet_2_2d_2_3d, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        planes = self.inplanes
        self.layer1 = self._make_layer(blocks[0], 64, layers[0])
        self.layer2 = self._make_layer(blocks[1], 128, layers[1], stride=2)
        self.layer3_ = self._make_layer(blocks[2], 256, layers[2], stride=2)
        self.layer4_ = self._make_layer(blocks[3], 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 1000)
        self.fc_ = nn.Linear(512, 512)

        self.fc_o = nn.Linear(512, 1)
        self.fc_a = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block.block_type == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=(1, stride, stride), bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            elif block.block_type == '2d':
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else: assert False

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        self.fov_increase += block.fov_increase
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            self.fov_increase += block.fov_increase

        self.block_types += [block.block_type]

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.order == 'BCDHW':
            x.transpose_(1, 2)
        elif self.order != 'BDCHW':
            assert False, "unknown dim order"

        B = x.shape[0]
        S = x.shape[1]
        D = S

        y = x.reshape([B * S, x.shape[2], x.shape[3], x.shape[4]])

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)

        y = y.reshape([B, S, y.shape[1], y.shape[2], y.shape[3]])
        y = y.transpose(1,2)
        y = self.layer3_(y)
        y = self.layer4_(y)
        y = y.transpose(1,2)
        y = y.reshape([B * S, y.shape[2], y.shape[3], y.shape[4]])

        y = self.avgpool(y)

        x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])

        x = self.fc_(x)
        x = nn.functional.relu(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        return offset, angle


def resnet18_2_2d_2_3d(load=True, blocknames=['BB13', 'BB13'], **kwargs):
    blocks = [BasicBlock, BasicBlock]

    for bname in blocknames:
        if bname == 'BB13':
            blocks += [BasicBlock_1_3]
        elif bname == 'BB33':
            blocks += [BasicBlock_3_3]
        elif bname == 'BB35':
            blocks += [BasicBlock_3_5]
        elif bname == 'BB55':
            blocks += [BasicBlock_5_5]
        elif bname == 'BB57':
            blocks += [BasicBlock_5_7]
        else:
            assert False

    for b in blocks: print(b.block_name)
    model = ResNet_2_2d_2_3d(blocks, [2, 2, 2, 2], **kwargs)
    if load:
        print("Loading dict from modelzoo..")
        model.load_state_dict(model_zoo.load_url(R.model_urls['resnet18']), strict=False)

        for layer in model.layer3_:
            layer.conv1.reset_parameters()
            layer.conv2.reset_parameters()
            layer.bn1.reset_parameters()
            layer.bn2.reset_parameters()
        for layer in model.layer4_:
            layer.conv1.reset_parameters()
            layer.conv2.reset_parameters()
            layer.bn1.reset_parameters()
            layer.bn2.reset_parameters()

    return model, blocks
