from torchvision.models import resnet
from torch import nn, stack
import torch
import torch.nn.functional as F
from torch.utils import model_zoo
from resnet.DropBlock import DropBlock2D
from resnet.convlstm import *


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, increase_factor=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        planes = self.inplanes
        self.layer1 = self._make_layer(block, planes, layers[0])
        planes *= increase_factor
        self.layer2 = self._make_layer(block, planes, layers[1], stride=2)
        planes *= increase_factor
        self.layer3 = self._make_layer(block, planes, layers[2], stride=2)
        planes *= increase_factor
        self.layer4 = self._make_layer(block, planes, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(planes * block.expansion, num_classes)

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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class KalmanConvLSTMHead(nn.Module):

    def __init__(self, input_dims, dims, per_pixel=False):
        super(KalmanConvLSTMHead, self).__init__()

        self.dims = dims
        self.per_pixel = per_pixel

        self.projection_conv = nn.Sequential(
                nn.Conv2d(input_dims, dims, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(dims),
            )

        self.lstm_f = ConvLSTMCellGeneral(input_dim=dims,
                                          hidden_dim=dims,
                                          kernel_size=(3, 3),
                                          bias=True,
                                          activation='tanh')
        if per_pixel:
            self.lstm_Q = ConvLSTMCellGeneral(input_dim=dims,
                                              hidden_dim=dims,
                                              kernel_size=(3, 3),
                                              bias=True,
                                              activation='tanh')
            self.lstm_R = ConvLSTMCellGeneral(input_dim=dims,
                                              hidden_dim=dims,
                                              kernel_size=(3, 3),
                                              bias=True,
                                              activation='tanh')
        else:
            self.lstm_Q = nn.LSTM(input_size=dims, hidden_size=dims, num_layers=1, batch_first=True)
            self.lstm_R = nn.LSTM(input_size=dims, hidden_size=dims, num_layers=1, batch_first=True)

        self.fc_f = nn.Linear(dims, dims)
        self.fc_F = nn.Linear(dims, dims)
        self.fc_Q = nn.Linear(dims, dims)
        self.fc_R = nn.Linear(dims, dims)

        self.P_init = nn.Parameter(0.1 * torch.eye(dims), requires_grad=True)
        self.I = nn.Parameter(torch.eye(dims), requires_grad=False)

        self.lstm_init = nn.Parameter(torch.zeros(dims).type(torch.Tensor), requires_grad=False)

        self.fc = nn.Linear(dims, dims)
        self.fc_o = nn.Linear(dims, 1)
        self.fc_a = nn.Linear(dims, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, get_features=False):

        B = x.shape[0]
        S = x.shape[1]
        C = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]
        D = self.dims

        y = x.reshape([B * S, C, H, W])
        y = self.projection_conv(y)
        x = y.reshape([B, S, y.shape[1], y.shape[2], y.shape[3]])
        C = x.shape[2]

        h_state_f = self.lstm_init.view(1, -1, 1, 1).expand(B, -1, H, W)
        c_state_f = self.lstm_init.view(1, -1, 1, 1).expand(B, -1, H, W)
        if self.per_pixel:
            h_state_Q = self.lstm_init.view(1, -1, 1, 1).expand(B, -1, H, W)
            c_state_Q = self.lstm_init.view(1, -1, 1, 1).expand(B, -1, H, W)
            h_state_R = self.lstm_init.view(1, -1, 1, 1).expand(B, -1, H, W)
            c_state_R = self.lstm_init.view(1, -1, 1, 1).expand(B, -1, H, W)
            P = self.P_init.view(1, 1, 1, self.dims, self.dims).expand(B, H, W, self.dims, self.dims)  # B, H, W, D, D
            I = self.I.view(1, 1, 1, self.dims, self.dims).expand(B, H, W, self.dims, self.dims)  # B, H, W, D, D

        else:
            h_state_Q = self.lstm_init.view(1, 1, -1).expand(1, B, -1).contiguous()
            c_state_Q = self.lstm_init.view(1, 1, -1).expand(1, B, -1).contiguous()
            h_state_R = self.lstm_init.view(1, 1, -1).expand(1, B, -1).contiguous()
            c_state_R = self.lstm_init.view(1, 1, -1).expand(1, B, -1).contiguous()
            P = self.P_init.view(1, self.dims, self.dims).expand(B, self.dims, self.dims)
            I = self.I.view(1, self.dims, self.dims).expand(B, self.dims, self.dims)
        y_hat = x[:, 0]
        y_outputs = []

        for s in range(S):

            lstm_f_out_, h_state_f, c_state_f = self.lstm_f(y_hat, (h_state_f, c_state_f))
            lstm_f_out = lstm_f_out_.permute(0, 2, 3, 1)  # B, H, W, D
            y_tick = self.fc_f(lstm_f_out).permute(0, 3, 1, 2)

            if self.per_pixel:
                F = self.fc_F(lstm_f_out)
                F1 = F.view(B, H, W, D, 1)  # .permute(0, 3, 4, 1, 2)   # B, H, W, D, 1
                F2 = F.view(B, H, W, 1, D)  # .permute(0, 3, 4, 1, 2)   # B, H, W, 1, D
                FF = torch.matmul(F1, F2)  # B, H, W, D, D

                lstm_Q_out_, h_state_Q, c_state_Q = self.lstm_Q(y_tick, (h_state_Q, c_state_Q))
                lstm_Q_out = lstm_Q_out_.permute(0, 2, 3, 1)  # B, H, W, D
                Q = matrix_diag(torch.exp(self.fc_Q(lstm_Q_out)))

                lstm_R_out_, h_state_R, c_state_R = self.lstm_R(x[:, s], (h_state_R, c_state_R))
                lstm_R_out = lstm_R_out_.permute(0, 2, 3, 1)  # B, H, W, D
                R = matrix_diag(torch.exp(self.fc_R(lstm_R_out)))

                P = P*FF + Q
                T = P+R
                K = torch.matmul(P, torch.inverse(T)) # B, H, W, D, D
                K_ = K

                I_KH = I - K
                P = torch.matmul(I_KH, torch.matmul(P, torch.transpose(I_KH, 3, 4))) + \
                    torch.matmul(K, torch.matmul(R, torch.transpose(K, 3, 4)))
            else:
                lstm_f_out_avg = self.avgpool(lstm_f_out_).view(B, -1)
                # print(lstm_f_out.shape)
                # print(lstm_f_out_avg.shape)
                F = self.fc_F(lstm_f_out_avg)
                F1 = F.view(B, D, 1)  # .permute(0, 3, 4, 1, 2)   # B, H, W, D, 1
                F2 = F.view(B, 1, D)  # .permute(0, 3, 4, 1, 2)   # B, H, W, 1, D
                FF = torch.matmul(F1, F2)  # B, H, W,

                y_tick_avg = self.avgpool(y_tick).view(B, 1, -1)
                lstm_Q_out, (h_state_Q, c_state_Q) = self.lstm_Q(y_tick_avg, (h_state_Q, c_state_Q))
                Q = matrix_diag(torch.exp(self.fc_Q(lstm_Q_out.view(B, -1))))

                x_avg = self.avgpool(x[:, s]).view(B, 1, -1)
                lstm_R_out, (h_state_R, c_state_R) = self.lstm_R(x_avg, (h_state_R, c_state_R))
                R = matrix_diag(torch.exp(self.fc_R(lstm_R_out.view(B, -1))))

                P = P*FF + Q
                T = P+R
                K = torch.matmul(P, torch.inverse(T))
                K_ = K.view(B, 1, 1, self.dims, self.dims).expand(B, H, W, self.dims, self.dims)  # B, H, W, D, D

                I_KH = I - K
                P = torch.matmul(I_KH, torch.matmul(P, torch.transpose(I_KH, 1, 2))) + \
                    torch.matmul(K, torch.matmul(R, torch.transpose(K, 1, 2)))

            v = x[:, 0] - y_tick

            y_hat_ = y_tick.permute(0, 2, 3, 1).view(B, H, W, D, 1) + torch.matmul(K_, v.permute(0, 2, 3, 1).view(B, H, W, D, 1))
            y_hat = y_hat_.view(B, H, W, D).permute(0, 3, 1, 2)
            y_outputs.append(y_hat)




        y = torch.stack(y_outputs, dim=1)
        C = y.shape[2]
        y = y.reshape([B * S, C, H, W])

        y = self.avgpool(y)

        x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])

        x = self.fc(x)
        x = nn.functional.relu(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        if get_features:
            assert False
            return offset, angle, x
        else:
            return offset, angle


class ConvLSTMHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, train_init, lstm_mem, relu_lstm, batchnorm, skip, bias, peephole, depth,
                 ar, h_skip, skip2, simple_skip, layernorm, leakyrelu):
        super(ConvLSTMHead, self).__init__()

        self.lstm_mem = lstm_mem
        self.ar = ar

        assert not batchnorm

        # if not relu_lstm:
        #     self.conv_lstm = ConvLSTMCell(input_dim=input_dim,
        #                                   hidden_dim=hidden_dim,
        #                                   kernel_size=(3, 3), bias=bias)
        # else:
        #     if batchnorm:
        #         self.conv_lstm = ConvLSTMCellReLUBN(input_dim=input_dim,
        #                                           hidden_dim=hidden_dim,
        #                                           kernel_size=(3, 3), bias=bias)
        #     elif skip:
        #         self.conv_lstm = ConvLSTMCellReLUSkip(input_dim=input_dim,
        #                                           hidden_dim=hidden_dim,
        #                                           kernel_size=(3, 3), bias=bias)
        #     else:
        #         self.conv_lstm = ConvLSTMCellReLU(input_dim=input_dim,
        #                                           hidden_dim=hidden_dim,
        #                                           kernel_size=(3, 3), bias=bias)

        self.conv_lstm = ConvLSTMCellGeneral(input_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             kernel_size=(3, 3),
                                             bias=bias,
                                             activation='leakyrelu' if leakyrelu else 'tanh',
                                             peephole=peephole,
                                             skip=skip, h_skip=h_skip, skip2=skip2, simple_skip=simple_skip,
                                             layernorm=layernorm, )

        self.conv_lstm_list = [self.conv_lstm]

        if depth > 1:
            for di in range(depth-1):
                # conv_lstm = ConvLSTMCellGeneral(input_dim=input_dim,
                conv_lstm = ConvLSTMCellGeneral(input_dim=self.conv_lstm_list[-1].output_dim,
                                                 hidden_dim=hidden_dim,
                                                 kernel_size=(3, 3),
                                                 bias=bias,
                                                 activation='leakyrelu' if leakyrelu else 'tanh',
                                                 peephole=peephole,
                                                 skip=skip, h_skip=h_skip, skip2=skip2, simple_skip=simple_skip,
                                                layernorm=layernorm)

                self.add_module('conv_lstm_%d' % (di+2), conv_lstm)

                self.conv_lstm_list += [conv_lstm]

        self.fc = nn.Linear(self.conv_lstm.output_dim, input_dim)
        self.fc_o = nn.Linear(input_dim, 1)
        self.fc_a = nn.Linear(input_dim, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm_init_list_h = []
        self.lstm_init_list_c = []

        self.train_init = train_init

        if train_init:
            self.lstm_init_h = nn.Parameter(torch.normal(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                                         0.01 * torch.ones(self.conv_lstm.hidden_dim).type(
                                                             torch.Tensor)), requires_grad=True)
            self.lstm_init_c = nn.Parameter(torch.normal(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                                         0.01 * torch.ones(self.conv_lstm.hidden_dim).type(
                                                             torch.Tensor)), requires_grad=True)

            self.lstm_init_list_h += [self.lstm_init_h]
            self.lstm_init_list_c += [self.lstm_init_c]

            if depth > 1:
                for di in range(depth-1):
                    lstm_init_h = nn.Parameter(torch.normal(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                                             0.01 * torch.ones(self.conv_lstm.hidden_dim).type(
                                                                 torch.Tensor)), requires_grad=True)
                    lstm_init_c = nn.Parameter(torch.normal(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                                             0.01 * torch.ones(self.conv_lstm.hidden_dim).type(
                                                                 torch.Tensor)), requires_grad=True)
                    self.lstm_init_list_h += [lstm_init_h]
                    self.lstm_init_list_c += [lstm_init_c]
                    self.register_parameter('conv_lstm_init_h_%d' % (di+1), lstm_init_h)
                    self.register_parameter('conv_lstm_init_c_%d' % (di+1), lstm_init_c)

        else:
            self.lstm_init_h = nn.Parameter(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                            requires_grad=False)
            self.lstm_init_c = nn.Parameter(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
                                            requires_grad=False)

        if ar:
            self.ar_init_f = nn.Parameter(torch.normal(torch.zeros(input_dim).type(torch.Tensor),
                                          0.1 * torch.ones(input_dim).type(torch.Tensor)),
                                          requires_grad=True)
            self.ar_init_o = nn.Parameter(torch.normal(torch.zeros(1).type(torch.Tensor),
                                          0.1 * torch.ones(1).type(torch.Tensor)),
                                          requires_grad=True)
            self.ar_init_a = nn.Parameter(torch.normal(torch.zeros(1).type(torch.Tensor),
                                          0.1 * torch.ones(1).type(torch.Tensor)),
                                          requires_grad=True)

    def forward(self, x, get_features=False):

        B = x.shape[0]
        S = x.shape[1]
        C = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]
        #
        # h_state = self.lstm_init_h.view(1, -1, 1, 1).expand(B, -1, H, W)
        # c_state = self.lstm_init_c.view(1, -1, 1, 1).expand(B, -1, H, W)

        y_outputs = []
        for d, conv_lstm in enumerate(self.conv_lstm_list):
            h_states = []

            if self.train_init:
                h_state = self.lstm_init_list_h[d].view(1, -1, 1, 1).expand(B, -1, H, W)
                c_state = self.lstm_init_list_c[d].view(1, -1, 1, 1).expand(B, -1, H, W)
            else:
                h_state = self.lstm_init_h.view(1, -1, 1, 1).expand(B, -1, H, W)
                c_state = self.lstm_init_c.view(1, -1, 1, 1).expand(B, -1, H, W)


            for s in range(S):

                if self.lstm_mem > 0 and s % self.lstm_mem == 0:
                    if self.train_init:
                        h_state = self.lstm_init_list_h[d].view(1, -1, 1, 1).expand(B, -1, H, W)
                        c_state = self.lstm_init_list_c[d].view(1, -1, 1, 1).expand(B, -1, H, W)
                    else:
                        h_state = self.lstm_init_h.view(1, -1, 1, 1).expand(B, -1, H, W)
                        c_state = self.lstm_init_c.view(1, -1, 1, 1).expand(B, -1, H, W)

                    # h_state = self.lstm_init_h.view(1, -1, 1, 1).expand(B, -1, H, W)
                    # c_state = self.lstm_init_c.view(1, -1, 1, 1).expand(B, -1, H, W)

                    # c_state = torch.stack([torch.stack(
                    #     [torch.stack([self.lstm_init_c for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in
                    #     range(B)],
                    #     dim=0)
                    # h_state = torch.stack([torch.stack(
                    #     [torch.stack([self.lstm_init_h for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in
                    #     range(B)],
                    #     dim=0)

                # else:
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

        if self.ar:
            # print(x[:,1:,:].shape)
            # print(self.ar_init_f.view(1,1,-1).expand(B,-1,-1).shape)
            z = torch.cat([self.ar_init_f.view(1,1,-1).expand(B,-1,-1), x[:,1:,:]], dim=1)

            # x_norm = torch.norm(x, 2, -1, keepdim=True)
            # z_norm = torch.norm(z, 2, -1, keepdim=True)

            # print(x_norm.shape, x.shape)

            # x_normed = torch.div(x, x_norm)
            # z_normed = torch.div(z, z_norm)

            # print(z_normed.shape)

            # xz = torch.mul(x_normed, z_normed)
            # xz_dot = torch.sum(xz, -1, keepdim=True)

            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            xz_dot = cos(x, z).view(B,S,1)

            # print(xz.shape)
            # print(xz_dot.shape)

            # print(self.ar_init_o.view(1,-1).expand(B,-1).shape, offset[:,1:].shape)

            oz = torch.cat([self.ar_init_o.view(1,1,-1).expand(B,1,-1), offset[:,1:]], dim=1)
            az = torch.cat([self.ar_init_a.view(1,1,-1).expand(B,1,-1), offset[:,1:]], dim=1)

            # print(oz.shape)

            offset = torch.mul(oz, xz_dot) + torch.mul(offset, 1-xz_dot)
            angle = torch.mul(az, xz_dot) + torch.mul(angle, 1-xz_dot)


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

    def __init__(self, block, layers, finetune=True, regional_pool=None, use_fc=False, use_convlstm=False,
                 trainable_lstm_init=False, width=None, height=None, conv_lstm_skip=False, confidence=False,
                 lstm_mem=0, second_head=False, relu_lstm=False, second_head_fc=False, lstm_bn=False, lstm_skip=False,
                 lstm_bias=False, lstm_peephole=False, lstm_state_reduction=1., lstm_depth=1, ar=False, bn=True,
                 kalman=False, order=None, h_skip=False, lstm_skip2=False, lstm_simple_skip=False, layernorm=False,
                 lstm_leakyrelu=False):

        super().__init__(block, layers)

        self.fov_increase = 0

        if regional_pool is not None:
            assert False, "regional_pool: not implemented"
        if confidence:
            assert False, "confidence: not implemented"

        self.dropblock = DropBlock2D()

        self.fc = None
        self.fc_ = None
        # self.avgpool = nn.AvgPool2d((12, 40), stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1) if regional_pool is None else regional_pool)

        self.trainable_lstm_init = trainable_lstm_init
        self.conv_lstm_skip = conv_lstm_skip

        self.confidence = confidence

        self.lstm_mem = lstm_mem

        self.head = None
        self.head2 = None

        self.second_head = second_head

        planes = 512 if kalman else 512

        if regional_pool is not None:
            self.regional_pool = True
            self.rp_conv = nn.Conv2d(planes, planes, kernel_size=regional_pool, stride=1, padding=0,
                               bias=False)
            self.rp_bn = nn.BatchNorm2d(planes)
        else:
            self.regional_pool = False

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
            self.fc_ = nn.Linear(planes * (2 if conv_lstm_skip else 1), planes)
        # else:
        #     self.lstm_init_h = nn.Parameter(torch.zeros(1, planes).type(torch.Tensor), requires_grad=True)
        #     self.lstm_init_c = nn.Parameter(torch.zeros(1, planes).type(torch.Tensor), requires_grad=True)
        #
        #     self.lstm = nn.LSTM(input_size=planes,
        #                         hidden_size=int(planes/lstm_state_reduction), num_layers=1,
        #                         batch_first=True)

        # if confidence:
        #     self.conf_fc = nn.Linear(512 * block.expansion, 2)

        self.use_convlstm = use_convlstm

        if use_convlstm:

            if kalman:
                self.head = KalmanConvLSTMHead(planes, int(planes/lstm_state_reduction))
            else:
                self.head = ConvLSTMHead(input_dim=planes,
                                         hidden_dim=int(planes/lstm_state_reduction),
                                         train_init=trainable_lstm_init,
                                         lstm_mem=lstm_mem, relu_lstm=relu_lstm, batchnorm=lstm_bn, skip=lstm_skip,
                                         bias=lstm_bias, peephole=lstm_peephole, depth=lstm_depth, ar=ar, h_skip=h_skip,
                                         skip2=lstm_skip2, simple_skip=lstm_simple_skip, layernorm=layernorm,
                                         leakyrelu=lstm_leakyrelu)

            if second_head:
                if second_head_fc:
                    self.head2 = FCHead(input_dim=planes,
                                        output_dim=planes)
                else:
                    self.head2 = ConvLSTMHead(input_dim=512*block.expansion,
                                              hidden_dim=int(512*block.expansion/lstm_state_reduction),
                                              train_init=trainable_lstm_init,
                                              lstm_mem=lstm_mem, relu_lstm=relu_lstm, batchnorm=lstm_bn, skip=lstm_skip,
                                              bias=lstm_bias, peephole=lstm_peephole, depth=lstm_depth, h_skip=h_skip,
                                               skip2=lstm_skip2, simple_skip=lstm_simple_skip, layernorm=layernorm,
                                         leakyrelu=lstm_leakyrelu)
            #
            # self.conv_lstm = ConvLSTMCell(input_dim=512*block.expansion,
            #                               hidden_dim=512*block.expansion,
            #                               kernel_size=(3,3), bias=False)
            # if self.trainable_lstm_init:
            #     self.lstm_init_h = nn.Parameter(torch.normal(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
            #                                     0.01*torch.ones(self.conv_lstm.hidden_dim).type(torch.Tensor))
            #                                     , requires_grad=True)
            #     self.lstm_init_c = nn.Parameter(torch.normal(torch.zeros(self.conv_lstm.hidden_dim).type(torch.Tensor),
            #                                     0.01*torch.ones(self.conv_lstm.hidden_dim).type(torch.Tensor))
            #                                     , requires_grad=True)
        else:
            self.head = FCHead(input_dim=512 * block.expansion,
                               output_dim=512 * block.expansion)

        # self.fc_o = nn.Linear(512 * block.expansion, 1)
        # self.fc_a = nn.Linear(512 * block.expansion, 1)

        # self.fc = self.fc_

    def set_dropblock_prob(self, prob):
        self.dropblock.keep_prob = prob

    def forward(self, x, use_dropblock=False, get_features=False):

        B = x.shape[0]
        S = x.shape[1]

        y = x.reshape([ B*S, x.shape[2], x.shape[3], x.shape[4] ])

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        # if use_dropblock:
        #     y = self.dropblock(y)
        y = self.layer4(y)
        # if use_dropblock:
        #     y = self.dropblock(y)

        C = y.shape[1]
        H = y.shape[2]
        W = y.shape[3]

        y = y.reshape([B, S, C, H, W])

        result = self.head(y, get_features)

        if self.second_head:
            offset2, angle2 = self.head2(y)

            return offset2, angle2, offset, angle

        else:
            return result

        # C = y.shape[1]
        # H = y.shape[2]
        # W = y.shape[3]

        # skip = y

        # if self.use_convlstm:
        #     y = y.reshape([B, S, C, H, W])
        #     h_states = []
        #     if self.trainable_lstm_init:
        #         c_state = torch.stack([torch.stack(
        #             [torch.stack([self.lstm_init_c for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in
        #                               range(B)], dim=0)
        #         h_state = torch.stack([torch.stack(
        #             [torch.stack([self.lstm_init_h for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in
        #                               range(B)], dim=0)
        #     else:
        #         h_state, c_state = self.conv_lstm.init_hidden(B, H, W)
        #
        #     for s in range(S):
        #
        #         if self.lstm_mem > 0 and s % self.lstm_mem == 0:
        #             h_state, c_state = self.conv_lstm.init_hidden(B, H, W)
        #
        #             context = torch.zeros(y[:,s,:,:,:].shape).type(torch.Tensor)#.cuda()
        #
        #             h_state, c_state = self.conv_lstm(torch.cat([y[:,s,:,:,:], context], dim=1), (h_state, c_state))
        #
        #         else:
        #             h_state, c_state = self.conv_lstm(y[:,s,:,:,:], (h_state, c_state))
        #         h_states.append(h_state)
        #
        #     # exit(0)
        #
        #     y = torch.stack(h_states, dim=1)
        #     y = y.reshape([B*S, C, H, W])
        #
        # if self.conv_lstm_skip:
        #     y = torch.cat([y, skip], dim=1)
        #
        # y = self.avgpool(y)
        #
        # x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])
        #
        # if self.use_fc:
        #     x = self.fc_(x)
        #     x = self.relu(x)
        # else:
        #
        #     init_c = torch.stack([self.lstm_init_c for _ in range(B)], dim=1)
        #     init_h = torch.stack([self.lstm_init_h for _ in range(B)], dim=1)
        #
        #     x, _ = self.lstm(x, (init_h, init_c))
        #
        # offset = self.fc_o(x)
        # angle = self.fc_a(x)


    def forward_convs_single(self, x):
        x = x.unsqueeze(0)

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        return y

    def forward_fcs_single_offset(self, x):

        B = x.shape[0]

        y = self.avgpool(x)

        x = y.reshape([B, 1, y.shape[1] * y.shape[2] * y.shape[3]])

        init_c = torch.stack([self.lstm_init_c for _ in range(1)], dim=1)
        init_h = torch.stack([self.lstm_init_h for _ in range(1)], dim=1)

        x, _ = self.lstm(x, (init_h, init_c))

        offset = self.fc_o(x)
        # angle = self.fc_a(x)

        return offset




def resnet18rnn(finetune=True, regional_pool=None, load=True, bn=True, **kwargs):
    model = ResNetPlusLSTM(resnet.BasicBlock if bn else BasicBlockNoBN, [2, 2, 2, 2],
                           finetune=finetune, regional_pool=regional_pool, **kwargs)
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

def resnet34rnn(finetune=False, regional_pool=None, load=True, bn=True, **kwargs):
    model = ResNetPlusLSTM(resnet.BasicBlock if bn else BasicBlockNoBN, [3, 4, 6, 3],
                           finetune=finetune, regional_pool=regional_pool, **kwargs)
    if load:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']), strict=False)
    return model

def resnet50rnn(finetune=False, regional_pool=None, **kwargs):
    model = ResNetPlusLSTM(resnet.Bottleneck, [3, 4, 6, 3], finetune=finetune, regional_pool=regional_pool, **kwargs)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']), strict=False)
    return model