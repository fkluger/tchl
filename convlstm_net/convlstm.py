import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvLSTMCellGeneral(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, activation='tanh', skip=False,
                 batch_norm=False, simple_skip=False):
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

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.skip = skip
        self.simple_skip = simple_skip
        self.batch_norm = batch_norm

        if activation == 'tanh':
            self.act_g = F.tanh
            self.act_c = F.tanh
        elif activation == 'relu':
            self.act_g = lambda x: x
            self.act_c = F.relu
        elif activation == 'leakyrelu':
            self.act_g = nn.LeakyReLU()
            self.act_c = nn.LeakyReLU()
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

        if skip and simple_skip:
            self.conv_y = nn.Conv2d(in_channels=self.hidden_dim,
                                    out_channels=self.input_dim,
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    bias=False)
            self.output_dim = self.input_dim

        elif skip:
            self.conv_y = nn.Conv2d(in_channels=self.input_dim + 2 * self.hidden_dim,
                                    out_channels=self.input_dim,
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    bias=False)
            self.output_dim = self.input_dim
            self.bn = nn.BatchNorm2d(self.input_dim)

        else:
            self.output_dim = self.hidden_dim

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        h_and_x = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        i = torch.sigmoid(self.conv_i(h_and_x))
        f = torch.sigmoid(self.conv_f(h_and_x))
        o = torch.sigmoid(self.conv_o(h_and_x))

        g = self.act_g(self.conv_g(h_and_x))

        c_next = f * c_cur + i * g

        if self.skip and self.simple_skip:
            h_next = o * self.act_c(c_next)
            y = self.conv_y(h_next) + input_tensor

        elif self.skip:
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
