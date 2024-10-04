import torch
from torch import nn
from torch import Tensor

class L1_LOSS(nn.Module):
    def __init__(self) -> None:
        super(L1_LOSS, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        _, _, h, w = input.size()
        norm = torch.sum(torch.abs(input), axis=(2, 3))
        norm = norm.div(h*w)
        return norm.mean()