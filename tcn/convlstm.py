import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import math


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next, h_next

    def init_hidden(self, batch_size, height, width):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())
    #
    # def init_hidden(self, batch_size, height, width):
    #     return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)),
    #             Variable(torch.zeros(batch_size, self.hidden_dim, height, width)))


class ConvLSTMCellReLU(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellReLU, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.bn = nn.BatchNorm2d(4 * self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        # combined_conv = self.bn(combined_conv)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = cc_g

        c_next = f * c_cur + i * g
        h_next = o * F.relu(c_next)

        return h_next, c_next, h_next


class ConvLSTMCellReLUSkip(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellReLUSkip, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv_f = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_i = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_g = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_o = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_y = nn.Conv2d(in_channels=self.input_dim + 2*self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=(1,1),
                                padding=(0,0),
                                bias=self.bias)


        self.bn = nn.BatchNorm2d(4 * self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        h_and_x = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        i = torch.sigmoid(self.conv_i(h_and_x))
        f = torch.sigmoid(self.conv_f(h_and_x))
        o = torch.sigmoid(self.conv_o(h_and_x))
        g = self.conv_g(h_and_x)

        c_next = f * c_cur + i * g
        h_hat = o * c_next
        h_next = F.relu(h_hat)

        h_and_x_and_h_hat = torch.cat([input_tensor, h_cur, h_hat], dim=1)  # concatenate along channel axis
        y = F.relu(self.conv_y(h_and_x_and_h_hat))

        return h_next, c_next, y


class ConvLSTMCellGeneral(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, activation='tanh', peephole=False, skip=False,
                 batch_norm=False):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellGeneral, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.peephole = peephole
        self.skip = skip
        self.batch_norm = batch_norm

        if activation == 'tanh':
            self.act_g = F.tanh
            self.act_c = F.tanh
        elif activation == 'relu':
            self.act_g = lambda x: x
            self.act_c = F.relu
        else:
            assert False, 'unknown activation function'

        self.conv_f = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_i = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_g = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_o = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

        if bias:
            self.conv_f.bias.data.fill_(1)
            self.conv_i.bias.data.zero_()
            self.conv_g.bias.data.zero_()
            self.conv_o.bias.data.zero_()

        if skip:
            self.conv_y = nn.Conv2d(in_channels=self.input_dim + 2*self.hidden_dim,
                                    out_channels=self.input_dim,
                                    kernel_size=(1,1),
                                    padding=(0,0),
                                    bias=False)
            self.output_dim = self.input_dim
            # if batch_norm:
            self.bn = nn.BatchNorm2d(self.input_dim)

        if peephole:
            self.w_cf = nn.Parameter(torch.Tensor(self.hidden_dim, 1, 1))
            self.w_ci = nn.Parameter(torch.Tensor(self.hidden_dim, 1, 1))
            self.w_co = nn.Parameter(torch.Tensor(self.hidden_dim, 1, 1))
            stdv = 1. / math.sqrt(self.hidden_dim)
            self.w_cf.data.uniform_(-stdv, stdv)
            self.w_ci.data.uniform_(-stdv, stdv)
            self.w_co.data.uniform_(-stdv, stdv)


    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        h_and_x = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        if self.peephole:
            i = torch.sigmoid(self.conv_i(h_and_x) + c_cur * self.w_ci)
            f = torch.sigmoid(self.conv_f(h_and_x) + c_cur * self.w_cf)
            o = torch.sigmoid(self.conv_o(h_and_x) + c_cur * self.w_co)
        else:
            i = torch.sigmoid(self.conv_i(h_and_x))
            f = torch.sigmoid(self.conv_f(h_and_x))
            o = torch.sigmoid(self.conv_o(h_and_x))

        g = self.act_g(self.conv_g(h_and_x))

        c_next = f * c_cur + i * g

        if self.skip:
            h_hat = o * c_next
            h_next = self.act_c(h_hat)

            h_and_x_and_h_hat = torch.cat([input_tensor, h_cur, h_hat], dim=1)  # concatenate along channel axis
            y_hat = self.conv_y(h_and_x_and_h_hat)
            if self.batch_norm:
                y_hat = self.bn(y_hat)
            y = self.act_c(y_hat + input_tensor)
        else:
            h_next = self.act_c(o * c_next)
            y = h_next

        return h_next, c_next, y


class ConvLSTMBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, hidden_plane_reduction=4, kernel_size=(3,3),
                 bias=False, batch_norm=False, act_y=nn.Tanh(), act_g=nn.Tanh(), act_c=nn.Tanh()):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMBlock, self).__init__()

        # self.height, self.width = input_size
        self.inplanes = inplanes
        self.hidden_dim = int(planes/hidden_plane_reduction)
        self.planes = planes

        self.downsample = downsample
        self.stride = stride

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.batch_norm = batch_norm

        self.act_g = act_g
        self.act_c = act_c
        self.act_y = act_y

        self.conv_f = nn.Conv2d(in_channels=self.inplanes + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_i = nn.Conv2d(in_channels=self.inplanes + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_g = nn.Conv2d(in_channels=self.inplanes + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)
        self.conv_o = nn.Conv2d(in_channels=self.inplanes + self.hidden_dim,
                                out_channels=self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=self.bias)

        self.conv_y = nn.Conv2d(in_channels=self.inplanes + 2 * self.hidden_dim,
                                out_channels=self.planes,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                stride=self.stride,
                                bias=self.bias)

        self.bn = nn.BatchNorm2d(self.planes)
        self.lstm_init = nn.Parameter(torch.zeros(self.hidden_dim).type(torch.Tensor), requires_grad=False)

    def forward(self, input_tensor, cur_state, cuda=True):

        B = input_tensor.shape[0]
        C = input_tensor.shape[1]
        H = input_tensor.shape[2]
        W = input_tensor.shape[3]

        h_cur, c_cur = cur_state

        if h_cur is None:
            # h_cur = torch.stack([torch.stack([torch.stack(
            #     [self.lstm_init for _ in range(H)],
            #     dim=-1) for _ in range(W)], dim=-1) for _ in range(B)], dim=0)
            h_cur = self.lstm_init.view(1, -1, 1, 1).expand(B, -1, H, W)
            # if cuda:
            #     h_cur = h_cur.cuda()
        if c_cur is None:
            # c_cur = torch.stack([torch.stack([torch.stack(
            #     [self.lstm_init for _ in range(H)],
            #     dim=-1) for _ in range(W)], dim=-1) for _ in range(B)], dim=0)
            c_cur = self.lstm_init.view(1, -1, 1, 1).expand(B, -1, H, W)
            # if cuda:
            #     c_cur = c_cur.cuda()

        residual = input_tensor

        h_and_x = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        i = torch.sigmoid(self.conv_i(h_and_x))
        f = torch.sigmoid(self.conv_f(h_and_x))
        o = torch.sigmoid(self.conv_o(h_and_x))

        g = self.act_g(self.conv_g(h_and_x))

        c_next = f * c_cur + i * g

        h_hat = o * c_next
        h_next = self.act_c(h_hat)

        h_and_x_and_h_hat = torch.cat([input_tensor, h_cur, h_hat], dim=1)  # concatenate along channel axis
        y_hat = self.conv_y(h_and_x_and_h_hat)
        if self.batch_norm:
            y_hat = self.bn(y_hat)

        if self.downsample is not None:
            residual = self.downsample(input_tensor)

        y = self.act_y(y_hat + residual)

        return h_next, c_next, y




class ConvLSTMResNet(nn.Module):

    def __init__(self, block, layers, activation='tanh', batch_norm=False, hidden_plane_reduction=1., device=None):
        self.inplanes = 64
        self.device = device
        super(ConvLSTMResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # if activation == 'relu':
        #     self.act = nn.ReLU(inplace=True)
        # elif activation == 'tanh':
        #     self.act = nn.Tanh()
        # else:
        #     assert False
        self.act = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)


        self.activation = activation
        self.layers = layers
        self.batch_norm = batch_norm
        self.hidden_plane_reduction = hidden_plane_reduction

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.fc = nn.Linear(512 * block.expansion, 512 * block.expansion)
        self.fc_o = nn.Linear(512 * block.expansion, 1)
        self.fc_a = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, batch_norm=self.batch_norm,
                            hidden_plane_reduction=self.hidden_plane_reduction,
                            act_y=nn.ReLU(inplace=True), act_g=nn.Tanh(), act_c=nn.Tanh()))
        if self.device is not None:
            layers[-1].to(self.device)

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def forward(self, x):
        B = x.shape[0]
        S = x.shape[1]
        C = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]
        x = x.reshape([B * S, C, H, W])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.reshape([B, S, x.shape[1], x.shape[2], x.shape[3]])

        B = x.shape[0]
        S = x.shape[1]
        C = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]

        h_states = [None for _ in range(sum(self.layers))]
        c_states = [None for _ in range(sum(self.layers))]
        y_list = []

        for t in range(S):
            y = x[:, t, :, :, :]
            for i, layer in enumerate(self.layer1):
                h_state = h_states[i]
                c_state = c_states[i]
                h_state, c_state, y = layer(y, (h_state, c_state))
                h_states[i] = h_state
                c_states[i] = c_state
            for i, layer in enumerate(self.layer2):
                h_state = h_states[i + sum(self.layers[:1])]
                c_state = c_states[i + sum(self.layers[:1])]
                h_state, c_state, y = layer(y, (h_state, c_state))
                h_states[i + sum(self.layers[:1])] = h_state
                c_states[i + sum(self.layers[:1])] = c_state
            for i, layer in enumerate(self.layer3):
                h_state = h_states[i + sum(self.layers[:2])]
                c_state = c_states[i + sum(self.layers[:2])]
                h_state, c_state, y = layer(y, (h_state, c_state))
                h_states[i + sum(self.layers[:2])] = h_state
                c_states[i + sum(self.layers[:2])] = c_state
            for i, layer in enumerate(self.layer4):
                h_state = h_states[i + sum(self.layers[:3])]
                c_state = c_states[i + sum(self.layers[:3])]
                h_state, c_state, y = layer(y, (h_state, c_state))
                h_states[i + sum(self.layers[:3])] = h_state
                c_states[i + sum(self.layers[:3])] = c_state

            y_list += [y]

        y = torch.stack(y_list, dim=1)
        C = y.shape[2]
        y = y.reshape([B * S, y.shape[2], y.shape[3], y.shape[4]])

        y = self.avgpool(y)

        x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])

        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.relu(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        return offset, angle


def convlstmresnet9(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ConvLSTMResNet(ConvLSTMBlock, [1, 1, 1, 1], **kwargs)

    return model

class ConvLSTMCellReLUBN(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellReLUBN, self).__init__()

        # self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv1 = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.conv2 = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.bn1 = nn.BatchNorm2d(4 * self.hidden_dim)
        self.bn2 = nn.BatchNorm2d(4 * self.hidden_dim)
        self.bn3 = nn.BatchNorm2d(self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        conv_h = self.conv1(h_cur)
        conv_x = self.conv1(input_tensor)

        combined_conv = self.bn1(conv_h) + self.bn2(conv_x)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = self.bn3(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * F.relu(c_next)

        return h_next, c_next, h_next

    def init_hidden(self, batch_size, height, width):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())
    #
    # def init_hidden(self, batch_size, height, width):
    #     return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)),
    #             Variable(torch.zeros(batch_size, self.hidden_dim, height, width)))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param