from losses.fro_loss import Fro_LOSS
from losses.l1_loss import L1_LOSS
from losses.per_loss import Per_LOSS
from losses.ssim_loss import SSIM
import torch
from torch import nn
from torch.nn import functional as F
__all__ = ['Fro_LOSS', 'L1_LOSS', 'Per_LOSS', 'SSIM']

def grad(img):
    # Custom kernel for edge detection or gradient calculation
    kernel = torch.tensor([[1 / 8, 1 / 8, 1 / 8], 
                           [1 / 8, -1, 1 / 8], 
                           [1 / 8, 1 / 8, 1 / 8]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Move kernel to the same device as the input image
    kernel = kernel.to(img.device)
    
    # Apply the kernel to the input image using conv2d
    grad_output = F.conv2d(img, kernel, padding='same')

    return grad_output

class Fusion_loss(torch.nn.Module):
    def __init__(self, batch_size):
        super(Fusion_loss, self).__init__()
        self.ssim1 = SSIM()
        self.ssim2 = SSIM()
        
        self.per1 = Per_LOSS()
        self.per2 = Per_LOSS()
        
        self.batch_size = batch_size
        self.W1 = nn.Parameter(torch.zeros(batch_size, 1), requires_grad=True)
        self.W2 = nn.Parameter(torch.zeros(batch_size, 1), requires_grad=True)
        
        #grad_loss
        self.grad_loss1 = Fro_LOSS()
        self.grad_loss2 = Fro_LOSS()
        
        
    
    def forward(self, SOURCE1, SOURCE2, 
                S1_FEAS, S2_FEAS, F_FEAS,
                generated_img, is_training=True):
        
        ssim_loss1 = self.ssim1(SOURCE1, SOURCE2)
        ssim_loss2 = self.ssim2(SOURCE1, SOURCE2)
        # ssim_loss = torch.mean(self.W1 * ssim_loss1 + self.W2 * ssim_loss2)
        
        perloss_1 = 0
        perloss_2 = 0
        for S1, S2, F in zip(S1_FEAS, S2_FEAS, F_FEAS):
            perloss_1 += self.per1(F - S1)
            perloss_2 += self.per2(F - S2)

        perloss_1 = perloss_1 / len(S1_FEAS)
        perloss_2 = perloss_2 / len(S2_FEAS)
        # per_loss = torch.mean(self.W1 * perloss_1 + self.W2 * perloss_2)
        
        grad_loss1 = self.grad_loss1(grad(generated_img) - grad(SOURCE1))
        grad_loss2 = self.grad_loss2(grad(generated_img) - grad(SOURCE2))
        # grad_loss = torch.mean(self.W1 * grad_loss1 + self.W2 * grad_loss2)
        
        self.W1*(ssim_loss1 + perloss_1 + grad_loss1)
        self.W2*(ssim_loss2 + perloss_2 + grad_loss2)
        return 
    
