from torch.nn.modules.loss import _Loss
import torch
import math
import numpy as np


class SqrtL1Loss(_Loss):

    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean', delta=0.25):
        self.reduce = reduce
        self.delta = 0.25
        self.a = 2*math.sqrt(delta)
        self.b = -delta
        super(SqrtL1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        absdiff = torch.clamp(torch.abs(input - target), 0, 1000.) + 1e-10
        sqrt = self.a*torch.sqrt(absdiff)+self.b

        losses = torch.where(absdiff <= self.delta, absdiff, sqrt)

        if not (self.reduce == False):
            return torch.mean(losses)
        return losses


def calc_horizon_leftright(width, height):
    wh = 0.5 * width*1./height

    def f(offset, angle):
        term2 = wh * torch.tan(torch.clamp(angle, -math.pi/3., math.pi/3.))
        return offset + 0.5 + term2, offset + 0.5 - term2

    return f


def horizon_error(width, height):

    calc_hlr = calc_horizon_leftright(width, height)

    def f(estm_ang, estm_off, true_ang, true_off):
        errors = []

        for b in range(estm_ang.shape[0]):
            for s in range(estm_ang.shape[1]):

                offset = true_off[b,s].squeeze()
                offset_estm = estm_off[b,s].squeeze()
                angle = true_ang[b,s].squeeze()
                angle_estm = estm_ang[b,s].squeeze()

                ylt, yrt = calc_hlr(offset, angle)
                yle, yre = calc_hlr(offset_estm, angle_estm)

                err1 = np.abs((ylt-yle).cpu().detach().numpy())
                err2 = np.abs((yrt-yre).cpu().detach().numpy())

                err = np.maximum(err1, err2)
                errors += [err]

        return errors

    return f


class MaxErrorLoss(_Loss):
    def __init__(self, size_average=True, reduce=True, reduction='elementwise_mean', from_start=False):
        super(MaxErrorLoss, self).__init__(size_average, reduce, reduction)
        self.from_start = from_start

    def forward(self, input, target):

        S = input.shape[1]

        input_diffs = []
        target_diffs = []

        if self.from_start:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,0,:]]
                target_diffs += [target[:,s,:]-target[:,0,:]]
        else:
            for s in range(1,S):
                input_diffs += [input[:,s,:]-input[:,s-1,:]]
                target_diffs += [target[:,s,:]-target[:,s-1,:]]

        target_diffs = torch.stack(target_diffs, dim=1)
        input_diffs = torch.stack(input_diffs, dim=1)

        return F.mse_loss(input_diffs, target_diffs, reduction=self.reduction)