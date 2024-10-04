import torch
from torch import nn
from torch import Tensor

class Fro_LOSS(nn.Module):
    def __init__(self) -> None:
        super(Fro_LOSS, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        _, _, h, w = input.size()
        fro_norm = torch.square(torch.norm(input, p='fro', dim=(2, 3))).div(h*w)
        fro_norm = torch.mean(fro_norm)
        return fro_norm
