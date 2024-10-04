import torch
from torch import nn
from torch import Tensor

class Per_LOSS(nn.Module):
    def __init__(self) -> None:
        super(Per_LOSS, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        _, c, h, w = input.size()
        norm = torch.sum(torch.square(input), axis=(1, 2, 3))
        loss = norm.div(h*w*c)
        return loss