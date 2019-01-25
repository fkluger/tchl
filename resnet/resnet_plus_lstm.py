from torchvision.models import resnet
from torch import nn, stack
import torch
import torch.nn.functional as F
from torch.utils import model_zoo
from resnet.DropBlock import DropBlock2D
from resnet.convlstm import *

class ConvLSTMHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, train_init, lstm_mem, relu_lstm, batchnorm, skip, bias):
        super(ConvLSTMHead, self).__init__()

        self.lstm_mem = lstm_mem

        if not relu_lstm:
            self.conv_lstm = ConvLSTMCell(input_dim=input_dim,
                                          hidden_dim=hidden_dim,
                                          kernel_size=(3, 3), bias=bias)
        else:
            if batchnorm:
                self.conv_lstm = ConvLSTMCellReLUBN(input_dim=input_dim,
                                                  hidden_dim=hidden_dim,
                                                  kernel_size=(3, 3), bias=bias)
            elif skip:
                self.conv_lstm = ConvLSTMCellReLUSkip(input_dim=input_dim,
                                                  hidden_dim=hidden_dim,
                                                  kernel_size=(3, 3), bias=bias)
            else:
                self.conv_lstm = ConvLSTMCellReLU(input_dim=input_dim,
                                                  hidden_dim=hidden_dim,
                                                  kernel_size=(3, 3), bias=bias)

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, 1)
        self.fc_a = nn.Linear(hidden_dim, 1)

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

    def forward(self, x):

        B = x.shape[0]
        S = x.shape[1]
        C = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]

        h_states = []

        c_state = torch.stack([torch.stack(
            [torch.stack([self.lstm_init_c for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in range(B)],
            dim=0)
        h_state = torch.stack([torch.stack(
            [torch.stack([self.lstm_init_h for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in range(B)],
            dim=0)

        for s in range(S):

            if self.lstm_mem > 0 and s % self.lstm_mem == 0:
                # h_state, c_state = self.conv_lstm.init_hidden(B, H, W)

                # context = torch.zeros(x[:, s, :, :, :].shape).type(torch.Tensor)  # .cuda()
                c_state = torch.stack([torch.stack(
                    [torch.stack([self.lstm_init_c for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in
                    range(B)],
                    dim=0)
                h_state = torch.stack([torch.stack(
                    [torch.stack([self.lstm_init_h for _ in range(H)], dim=-1) for _ in range(W)], dim=-1) for _ in
                    range(B)],
                    dim=0)
                # h_state, c_state = self.conv_lstm(torch.cat([x[:, s, :, :, :], context], dim=1), (h_state, c_state))

            # else:
            h_state, c_state, y_step = self.conv_lstm(x[:, s, :, :, :], (h_state, c_state))
            h_states.append(y_step)

        y = torch.stack(h_states, dim=1)
        y = y.reshape([B * S, C, H, W])

        y = self.avgpool(y)

        x = y.reshape([B, S, y.shape[1] * y.shape[2] * y.shape[3]])

        # print(B, S, C, H, W)
        # print(x.shape)

        x = self.fc(x)
        x = nn.functional.relu(x)

        offset = self.fc_o(x)
        angle = self.fc_a(x)

        return offset, angle


class FCHead(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FCHead, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.fc_o = nn.Linear(output_dim, 1)
        self.fc_a = nn.Linear(output_dim, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):

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


class ResNetPlusLSTM(resnet.ResNet):

    def __init__(self, block, layers, finetune=False, regional_pool=None, use_fc=False, use_convlstm=False,
                 trainable_lstm_init=False, width=None, height=None, conv_lstm_skip=False, confidence=False,
                 lstm_mem=0, second_head=False, relu_lstm=False, second_head_fc=False, lstm_bn=False, lstm_skip=False,
                 lstm_bias=False):

        super().__init__(block, layers)

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

        if regional_pool is not None:
            self.regional_pool = True
            self.rp_conv = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=regional_pool, stride=1, padding=0,
                               bias=False)
            self.rp_bn = nn.BatchNorm2d(512 * block.expansion)
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
            self.fc_ = nn.Linear(512 * block.expansion * (2 if conv_lstm_skip else 1), 512 * block.expansion)
        else:
            self.lstm_init_h = nn.Parameter(torch.zeros(1, 512 * block.expansion).type(torch.Tensor), requires_grad=True)
            self.lstm_init_c = nn.Parameter(torch.zeros(1, 512 * block.expansion).type(torch.Tensor), requires_grad=True)

            self.lstm = nn.LSTM(input_size=512 * block.expansion, hidden_size=512 * block.expansion, num_layers=1,
                                batch_first=True)

        if confidence:
            self.conf_fc = nn.Linear(512 * block.expansion, 2)

        self.use_convlstm = use_convlstm

        if use_convlstm:

            self.head = ConvLSTMHead(input_dim=512*block.expansion,
                                     hidden_dim=512*block.expansion,
                                     train_init=trainable_lstm_init,
                                     lstm_mem=lstm_mem, relu_lstm=relu_lstm, batchnorm=lstm_bn, skip=lstm_skip, bias=lstm_bias)

            if second_head:
                if second_head_fc:
                    self.head2 = FCHead(input_dim=512*block.expansion,
                                        output_dim=512*block.expansion)
                else:
                    self.head2 = ConvLSTMHead(input_dim=512*block.expansion,
                                        hidden_dim=512*block.expansion,
                                         train_init=trainable_lstm_init,
                                         lstm_mem=lstm_mem, relu_lstm=relu_lstm, batchnorm=lstm_bn, skip=lstm_skip, bias=lstm_bias)
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

        # self.fc_o = nn.Linear(512 * block.expansion, 1)
        # self.fc_a = nn.Linear(512 * block.expansion, 1)

        # self.fc = self.fc_

    def set_dropblock_prob(self, prob):
        self.dropblock.keep_prob = prob

    def forward(self, x, use_dropblock=False):

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
        if use_dropblock:
            y = self.dropblock(y)
        y = self.layer4(y)
        if use_dropblock:
            y = self.dropblock(y)

        C = y.shape[1]
        H = y.shape[2]
        W = y.shape[3]

        y = y.reshape([B, S, C, H, W])

        offset, angle = self.head(y)

        if self.second_head:
            offset2, angle2 = self.head2(y)

            return offset2, angle2, offset, angle

        else:
            return offset, angle

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


def resnet18rnn(finetune=False, regional_pool=None, load=True, **kwargs):
    model = ResNetPlusLSTM(resnet.BasicBlock, [2, 2, 2, 2], finetune=finetune, regional_pool=regional_pool, **kwargs)
    if load:
        model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet18']), strict=False)
    return model

def resnet50rnn(finetune=False, regional_pool=None, **kwargs):
    model = ResNetPlusLSTM(resnet.Bottleneck, [3, 4, 6, 3], finetune=finetune, regional_pool=regional_pool, **kwargs)
    model.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet50']), strict=False)
    return model