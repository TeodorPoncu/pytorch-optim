from functools import partial
from torch import nn
import torch


def get_activation_function(type: str) -> partial:
    if type == 'relu':
        return partial(nn.ReLU, inplace=True)
    elif type == 'lrelu':
        return partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif type == 'sigmoid':
        return partial(nn.Sigmoid, inplace=True)
    elif type == 'tanh':
        return partial(nn.Tanh, inplace=True)
    else:
        return partial(IdentityLayer)


def get_normalization_function(type: str) -> partial:
    if type == 'batch':
        # be careful when using batch_norm as it does not work on a batch_size of 1
        return partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif type == 'layer':
        return partial(nn.LayerNorm)
    else:
        return partial(IdentityLayer)


class IdentityLayer(nn.Module):
    def __init__(self, *args):
        super(IdentityLayer, self).__init__()

    def forward(self, x) -> torch.Tensor:
        return x