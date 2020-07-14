import torch
from torch.nn.modules.module import Module


class MaxupCrossEntropyLoss(Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input, target, valid=False):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        batch_size = target.shape[0]
        target = target.reshape(input.shape)
        if valid is True:
            loss = torch.sum(-target * logsoftmax(input), dim=1)
        else:
            loss = torch.sum(-target * logsoftmax(input), dim=1)
            loss, _ = loss.reshape((batch_size, self.m)).max(1)
        loss = torch.mean(loss)
        return loss
