from torch.nn.modules.loss import _Loss
import torch
import math

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
