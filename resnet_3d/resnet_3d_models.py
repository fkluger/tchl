from torchvision.models import resnet as R
from torch import nn, stack
import torch
import torch.nn.functional as F
from torch.utils import model_zoo
import math
from resnet.convlstm import *

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConvLSTMHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, train_init, lstm_mem, relu_lstm, skip, bias, peephole, depth,
                 h_skip):
        super(ConvLSTMHead, self).__init__()

        self.lstm_mem = lstm_mem

        self.conv_lstm = ConvLSTMCellGeneral(input_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             kernel_size=(3, 3),
                                             bias=bias,
                                             activation='relu' if relu_lstm else 'tanh',
                                             peephole=peephole,
                                             skip=skip, h_skip=h_skip)

        self.conv_lstm_list = [self.conv_lstm]

        if depth > 1:
            conv_lstm = ConvLSTMCellGeneral(input_dim=input_dim,
                                            hidden_dim=hidden_dim,
                                            kernel_size=(3, 3),
                                            bias=bias,
                                            activation='relu' if relu_lstm else 'tanh',
                                            peephole=peephole,
                                            skip=skip, h_skip=h_skip)

            self.conv_lstm_list += [conv_lstm]

        self.fc = nn.Linear(self.conv_lstm.output_dim, input_dim)
        self.fc_o = nn.Linear(input_dim, 1)
        self.fc_a = nn.Linear(input_dim, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if train_init:
            self.lstm_init_h = nn.Parameter(torch.normal(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                                         0.01 * torch.ones(self.conv_lstm.hidden_dim).type(
                                                             torch.Tensor)), requires_grad=True)
            self.lstm_init_c = nn.Parameter(torch.normal(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                                         0.01 * torch.ones(self.conv_lstm.hidden_dim).type(
                                                             torch.Tensor)), requires_grad=True)
        else:
            self.lstm_init_h = nn.Parameter(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                            requires_grad=False)
            self.lstm_init_c = nn.Parameter(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                            requires_grad=False)


    def forward(self, x, get_features=False):

        B = x.shape[0]
        S = x.shape[1]
        C = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]

        h_state = self.lstm_init_h.view(1, -1, 1, 1).expand(B, -1, H, W)
        c_state = self.lstm_init_c.view(1, -1, 1, 1).expand(B, -1, H, W)

        y_outputs = []
        for d, conv_lstm in enumerate(self.conv_lstm_list):
            h_states = []
            for s in range(S):

                if self.lstm_mem > 0 and s % self.lstm_mem == 0:
                    c_state = torch.stack([torch.stack(
                        [torch.stack([self.lstm_init_c for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in
                        range(B)],
                        dim=0)
                    h_state = torch.stack([torch.stack(
                        [torch.stack([self.lstm_init_h for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in
                        range(B)],
                        dim=0)

                if d == 0:
                    h_state, c_state, y_step = conv_lstm(x[:, s, :, :, :], (h_state, c_state))
                else:
                    h_state, c_state, y_step = conv_lstm(y_outputs[-1][s], (h_state, c_state))
                h_states.append(y_step)

            y_outputs.append(h_states.copy())

        y = torch.stack(y_outputs[-1], dim=1)
        C = y.shape[2]
        y = y.reshape([B * S, C, H, W])

        y = self.avgpool(y)

        x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])

        x = self.fc(x)
        x = nn.functional.relu(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        if get_features:
            return offset, angle, x
        else:
            return offset, angle



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


class BasicBlock_1_3dil(nn.Module):
    expansion = 1
    fov_increase = 2
    block_type = '3d'
    block_name = "BasicBlock_1_3dil"

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_1_3dil, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                     padding=(1, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=(1, 1, 1),
                               padding=(2, 1, 1), dilation=(2, 1, 1), bias=False)
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

class BasicBlock_3_3dil(nn.Module):
    expansion = 1
    fov_increase = 3
    block_type = '3d'
    block_name = "BasicBlock_3_3dil"

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_3_3dil, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), stride=(1, stride, stride),
                     padding=(1, 1, 1), dilation=(1, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=(1, 1, 1),
                               padding=(2, 1, 1), dilation=(2, 1, 1), bias=False)
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

        # y = y.reshape([B, S, C, H, W])

        # y = x.reshape([B * S, C, H, W])

        y = self.avgpool(y)

        x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])

        x = self.fc_(x)
        x = nn.functional.relu(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        return offset, angle


class ResNetAR(nn.Module):

    def __init__(self, blocks, order='BCDHW'):
        self.block_types = []
        self.order = order
        self.fov_increase = 0
        self.inplanes = 64
        super(ResNetAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.softmax = torch.nn.Softmax(dim=-1)
        planes = self.inplanes
        self.layer1 = self._make_layer(blocks[0:2], 64)
        self.layer2 = self._make_layer(blocks[2:4], 128, stride=2)
        self.layer3 = self._make_layer(blocks[4:6], 256, stride=2)
        self.layer4 = self._make_layer(blocks[6:8], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv_end = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn_end = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, 1000)
        self.fc_ = nn.Linear(512, 512)

        self.fc_o = nn.Linear(512, 1)
        self.fc_a = nn.Linear(512, 1)

        # self.init_featmap = nn.Parameter(torch.zeros(512).type(torch.Tensor), requires_grad=False)

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

    def _make_layer(self, blocks, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * blocks[0].expansion:
            if blocks[0].block_type == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * blocks[0].expansion,
                              kernel_size=1, stride=(1, stride, stride), bias=False),
                    nn.BatchNorm3d(planes * blocks[0].expansion),
                )
            elif blocks[0].block_type == '2d':
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * blocks[0].expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * blocks[0].expansion),
                )
            else: assert False

        layers = []
        layers.append(blocks[0](self.inplanes, planes, stride, downsample))
        self.inplanes = planes * blocks[0].expansion
        self.fov_increase += blocks[0].fov_increase
        for i in range(1, len(blocks)):
            layers.append(blocks[i](self.inplanes, planes))
            self.fov_increase += blocks[i].fov_increase

        self.block_types += [blocks[0].block_type]

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

        y = y.reshape([B, S, y.shape[1], y.shape[2], y.shape[3]])
        # y = y.transpose(1,2)


        y_norms = torch.norm(y, dim=2, keepdim=True)
        y_normed = torch.div(y, y_norms + 1e-6)  # B, S, C, H, W
        # y_normed = y

        yn1 = y_normed.permute(0, 3, 4, 1, 2) # B, H, W, S, C
        yn2 = y_normed.permute(0, 3, 4, 2, 1) # B, H, W, C, S
        A = self.relu(torch.matmul(yn1, yn2))  # B, H, W, S, S

        yo_list = []
        for s in range(S):
            ys = y[:, s, :, :, :]
            cs = torch.zeros_like(ys)
            if s == 0:
                cs = torch.zeros_like(ys)
                yo = self.conv_end(torch.cat([ys, cs], dim=1))
            else:
                ai_list = self.softmax(A[:,:,:,0:s,s])

                for i in range(s):
                    yi = yo_list[i]  # y[:, i, :, :, :]
                    ai = ai_list[:,:,:,i].view(yi.shape[0], 1, yi.shape[2], yi.shape[3]).expand(-1, yi.shape[1], -1, -1)
                    cs = cs + yi*ai
                yo = self.conv_end(torch.cat([ys, cs], dim=1))
            yo_list += [yo]

        y = torch.stack(yo_list, dim=1)

        # y = y.transpose(1,2)
        y = y.reshape([B * S, y.shape[2], y.shape[3], y.shape[4]])

        y = self.bn_end(y)
        y = self.relu(y)

        y = self.avgpool(y)

        x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])

        x = self.fc_(x)
        x = nn.functional.relu(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        return offset, angle



class ResNet_3_2d_1_3d(nn.Module):

    def __init__(self, blocks, order='BCDHW', conv_lstm=False):
        self.block_types = []
        self.order = order
        self.fov_increase = 0
        self.inplanes = 64
        super(ResNet_3_2d_1_3d, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        planes = self.inplanes
        self.layer1 = self._make_layer(blocks[0:2], 64)
        self.layer2 = self._make_layer(blocks[2:4], 128, stride=2)
        self.layer3 = self._make_layer(blocks[4:6], 256, stride=2)
        self.layer4_ = self._make_layer(blocks[6:8], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, 1000)

        if conv_lstm:
            self.conv_lstm = ConvLSTMHead(input_dim=512,
                                          hidden_dim=128,
                                          train_init=False,
                                          lstm_mem=0, relu_lstm=False, skip=True,
                                          bias=False, peephole=False, depth=1, h_skip=False)
        else:
            self.conv_lstm = None
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

    def _make_layer(self, blocks, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * blocks[0].expansion:
            if blocks[0].block_type == '3d':
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * blocks[0].expansion,
                              kernel_size=1, stride=(1, stride, stride), bias=False),
                    nn.BatchNorm3d(planes * blocks[0].expansion),
                )
            elif blocks[0].block_type == '2d':
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * blocks[0].expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * blocks[0].expansion),
                )
            else: assert False

        layers = []
        layers.append(blocks[0](self.inplanes, planes, stride, downsample))
        self.inplanes = planes * blocks[0].expansion
        self.fov_increase += blocks[0].fov_increase
        for i in range(1, len(blocks)):
            layers.append(blocks[i](self.inplanes, planes))
            self.fov_increase += blocks[i].fov_increase

        self.block_types += [blocks[0].block_type]

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

        y = y.reshape([B, S, y.shape[1], y.shape[2], y.shape[3]])
        y = y.transpose(1,2)
        y = self.layer4_(y)
        y = y.transpose(1,2)

        if self.conv_lstm is not None:
            offset, angle = self.conv_lstm(y)
        else:

            y = y.reshape([B * S, y.shape[2], y.shape[3], y.shape[4]])

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


def resnet18(load=True, **kwargs):
    blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock]
    for b in blocks: print(b.block_name)
    model = ResNet3d(blocks, [2, 2, 2, 2], **kwargs)
    if load:
        print("Loading dict from modelzoo..")
        model.load_state_dict(model_zoo.load_url(R.model_urls['resnet18']), strict=False)

        for layer in model.layer3:
            layer.conv1.reset_parameters()
            layer.conv2.reset_parameters()
            layer.bn1.reset_parameters()
            layer.bn2.reset_parameters()
        for layer in model.layer4:
            layer.conv1.reset_parameters()
            layer.conv2.reset_parameters()
            layer.bn1.reset_parameters()
            layer.bn2.reset_parameters()

    return model, blocks

def resnet_ar(load=True, **kwargs):
    blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock, BasicBlock, BasicBlock, BasicBlock, BasicBlock]
    for b in blocks: print(b.block_name)
    model = ResNetAR(blocks, **kwargs)
    if load:
        print("Loading dict from modelzoo..")
        model.load_state_dict(model_zoo.load_url(R.model_urls['resnet18']), strict=False)

        for layer in model.layer3:
            layer.conv1.reset_parameters()
            layer.conv2.reset_parameters()
            layer.bn1.reset_parameters()
            layer.bn2.reset_parameters()
        for layer in model.layer4:
            layer.conv1.reset_parameters()
            layer.conv2.reset_parameters()
            layer.bn1.reset_parameters()
            layer.bn2.reset_parameters()

    return model, blocks

def resnet18_3_2d_1_3d(load=True, blocknames=None, **kwargs):
    lastblocks = []
    if blocknames is None:
        lastblocks += [BasicBlock, BasicBlock]
    else:
        for name in blocknames:
            if name == "BB11":
                lastblocks += [BasicBlock_1_1]
            elif name == "BB13":
                lastblocks += [BasicBlock_1_3]
            elif name == "BB33":
                lastblocks += [BasicBlock_3_3]
            elif name == "BB35":
                lastblocks += [BasicBlock_3_5]
            elif name == "BB55":
                lastblocks += [BasicBlock_5_5]
            elif name == "BB57":
                lastblocks += [BasicBlock_5_7]
            elif name == "BB13d":
                lastblocks += [BasicBlock_1_3dil]
            elif name == "BB33d":
                lastblocks += [BasicBlock_3_3dil]
            else:
                assert False

    blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock, BasicBlock, BasicBlock, lastblocks[0], lastblocks[1]]

    for b in blocks: print(b.block_name)
    model = ResNet_3_2d_1_3d(blocks, **kwargs)
    if load:
        print("Loading dict from modelzoo..")
        model.load_state_dict(model_zoo.load_url(R.model_urls['resnet18']), strict=False)

        for layer in model.layer3:
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


def resnet18_3_2d_1_3d_lstm(load=True, **kwargs):
    blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock, BasicBlock, BasicBlock, BasicBlock_1_3, BasicBlock_3_3]
    for b in blocks: print(b.block_name)
    model = ResNet_3_2d_1_3d(blocks, conv_lstm=True, **kwargs)
    if load:
        print("Loading dict from modelzoo..")
        model.load_state_dict(model_zoo.load_url(R.model_urls['resnet18']), strict=False)

        for layer in model.layer3:
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

def resnet18_2_2d_2_3d(load=True, **kwargs):
    blocks = [BasicBlock, BasicBlock, BasicBlock_1_3, BasicBlock_3_3]
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


def resnet18_3d_3_3(load=True, **kwargs):
    model = ResNet3d([BasicBlock_3_3, BasicBlock_3_3, BasicBlock_3_3, BasicBlock_3_3], [2, 2, 2, 2], **kwargs)
    return model

def resnet18_2d3d_3_3dil(load=True, **kwargs):
    model = ResNet3d([BasicBlock, BasicBlock, BasicBlock_3_3dil, BasicBlock_3_3dil], [2, 2, 2, 2], **kwargs)
    if load:
        print("Loading dict from modelzoo..")        # model.load_state_dict(model_zoo.load_url(R.model_urls['resnet18']), strict=False)

        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(R.model_urls['resnet18'])

        forbidden_layers = ['layer3', 'layer4', 'fc']

        pruned_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            forbidden = False
            for fl in forbidden_layers:
                if fl in k:
                    forbidden = True
            if not forbidden:
                pruned_pretrained_dict[k] = v

        model_dict.update(pruned_pretrained_dict)
        model.load_state_dict(model_dict)

    return model

def resnet18_2d(load=True, **kwargs):
    model = ResNet3d([BasicBlock, BasicBlock, BasicBlock, BasicBlock], [2, 2, 2, 2], **kwargs)
    if load:
        print("Loading dict from modelzoo..")        # model.load_state_dict(model_zoo.load_url(R.model_urls['resnet18']), strict=False)

        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(R.model_urls['resnet18'])

        forbidden_layers = ['layer3', 'layer4', 'fc']

        pruned_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            forbidden = False
            for fl in forbidden_layers:
                if fl in k:
                    forbidden = True
            if not forbidden:
                pruned_pretrained_dict[k] = v

        model_dict.update(pruned_pretrained_dict)
        model.load_state_dict(model_dict)

        for layer in model.layer3:
            layer.conv1.reset_parameters()
            layer.conv2.reset_parameters()
            layer.bn1.reset_parameters()
            layer.bn2.reset_parameters()
        for layer in model.layer4:
            layer.conv1.reset_parameters()
            layer.conv2.reset_parameters()
            layer.bn1.reset_parameters()
            layer.bn2.reset_parameters()

    return model

def resnet18_3d_1_3dil(**kwargs):
    model = ResNet3d(BasicBlock_1_3dil, [2, 2, 2, 2], **kwargs)
    return model
#
# def resnet34rnn(finetune=False, regional_pool=None, load=True, bn=True, **kwargs):
#     model = ResNetPlusLSTM(resnet.BasicBlock if bn else BasicBlockNoBN, [3, 4, 6, 3],
#                            finetune=finetune, regional_pool=regional_pool, **kwargs)
#     if load:
#         model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']), strict=False)
#     return model
#
# def resnet50rnn(finetune=False, regional_pool=None, **kwargs):
#     model = ResNetPlusLSTM(resnet.Bottleneck, [3, 4, 6, 3], finetune=finetune, regional_pool=regional_pool, **kwargs)
#     model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']), strict=False)
#     return model