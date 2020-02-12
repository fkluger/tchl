from torchvision.models import resnet
import torch.nn.functional as F
from torch.utils import model_zoo
from convlstm_net.convlstm import *


class ConvLSTMHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, skip, bias, depth,
                 simple_skip):
        super(ConvLSTMHead, self).__init__()

        self.conv_lstm = ConvLSTMCellGeneral(input_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             kernel_size=(3, 3),
                                             bias=bias,
                                             activation='tanh',
                                             skip=skip, simple_skip=simple_skip)

        self.conv_lstm_list = [self.conv_lstm]

        if depth > 1:
            for di in range(depth - 1):
                conv_lstm = ConvLSTMCellGeneral(input_dim=self.conv_lstm_list[-1].output_dim,
                                                hidden_dim=hidden_dim,
                                                kernel_size=(3, 3),
                                                bias=bias,
                                                activation='tanh',
                                                skip=skip, simple_skip=simple_skip)

                self.add_module('conv_lstm_%d' % (di + 2), conv_lstm)

                self.conv_lstm_list += [conv_lstm]

        self.fc = nn.Linear(self.conv_lstm.output_dim, input_dim)
        self.fc_o = nn.Linear(input_dim, 1)
        self.fc_a = nn.Linear(input_dim, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm_init_list_h = []
        self.lstm_init_list_c = []

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

        y_outputs = []
        for d, conv_lstm in enumerate(self.conv_lstm_list):
            h_states = []

            h_state = self.lstm_init_h.view(1, -1, 1, 1).expand(B, -1, H, W)
            c_state = self.lstm_init_c.view(1, -1, 1, 1).expand(B, -1, H, W)

            for s in range(S):

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


class FCHead(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FCHead, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.fc_o = nn.Linear(output_dim, 1)
        self.fc_a = nn.Linear(output_dim, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, get_features=False):
        B = x.shape[0]
        S = x.shape[1]
        C = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]

        y = x.reshape([B * S, C, H, W])

        y = self.avgpool(y)

        x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])

        x = self.fc(x)
        x = nn.functional.relu(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        return offset, angle


class BasicBlockNoBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockNoBN, self).__init__()
        self.conv1 = resnet.conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resnet.conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetPlusLSTM(resnet.ResNet):

    def __init__(self, block, layers, finetune=True, use_fc=False, use_convlstm=False, lstm_skip=False,
                 lstm_bias=False, lstm_state_reduction=1., lstm_depth=1,
                 lstm_simple_skip=False):

        super().__init__(block, layers)

        self.fov_increase = 0

        self.fc = None
        self.fc_ = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = None
        self.head2 = None

        planes = 512

        if not finetune:
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False

        self.use_fc = use_fc

        if use_fc:
            self.fc_ = nn.Linear(planes, planes)

        self.use_convlstm = use_convlstm

        if use_convlstm:
            self.head = ConvLSTMHead(input_dim=planes, hidden_dim=int(planes / lstm_state_reduction), skip=lstm_skip,
                                     bias=lstm_bias, depth=lstm_depth, simple_skip=lstm_simple_skip)

        else:
            self.head = FCHead(input_dim=512 * block.expansion,
                               output_dim=512 * block.expansion)

    def forward(self, x, get_features=False):

        B = x.shape[0]
        S = x.shape[1]

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

        y = y.reshape([B, S, C, H, W])

        result = self.head(y, get_features)

        return result


def resnet18rnn(finetune=True, load=True, bn=True, **kwargs):
    model = ResNetPlusLSTM(resnet.BasicBlock if bn else BasicBlockNoBN, [2, 2, 2, 2], finetune=finetune, **kwargs)
    if load:
        print("Loading dict from modelzoo..")
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']), strict=False)

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
