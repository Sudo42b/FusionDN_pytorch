import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
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
        

class Per_LOSS(nn.Module):
    def __init__(self) -> None:
        super(Per_LOSS, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        _, c, h, w = input.size()
        norm = torch.sum(torch.square(input), axis=(1, 2, 3))
        loss = norm.div(h*w*c)
        return loss

class Fro_LOSS(nn.Module):
    def __init__(self) -> None:
        super(Fro_LOSS, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        _, _, h, w = input.size()
        fro_norm = torch.square(torch.norm(input, p='fro', dim=(2, 3))).div(h*w)
        fro_norm = torch.mean(fro_norm)
        return fro_norm


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    """Reference
                https://ece.uwaterloo.ca/~z70wang/research/ssim/
    """
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

if __name__ == "__main__":
    l1 = L1_LOSS()
    per = Per_LOSS()
    fro = Fro_LOSS()
    x1 = torch.randn([16, 3, 256, 256])
    x2 = torch.randn([16, 3, 256, 256])
    sim = SSIM()
    