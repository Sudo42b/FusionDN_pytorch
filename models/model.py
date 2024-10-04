import torch
from torch import nn
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from .vgg16 import get_vgg16
from .generator import Generator

class ResizeConcat(nn.Module):
    def __init__(self, output_size):
        super(ResizeConcat, self).__init__()
        self.output_size = output_size

    def forward(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = resize(image, (new_h, new_w), interpolation=InterpolationMode.BILINEAR)
        if len(img.shape) == 3:
            img = img.unsqueeze(1)
        return torch.cat([img, img, img], dim=1)

class MODEL(nn.Module):
    def __init__(self, batch_size, 
                 patch_size=64,
                 input_size=224, ):#Image_size
        super(MODEL, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.path_size = patch_size
        self.generator = Generator()
        
        self.vgg1 = get_vgg16()
        self.vgg2 = get_vgg16()
        self.vggF = get_vgg16()
        self.resized_concat = ResizeConcat((input_size, input_size))
        
    def forward(self, I1, I2):
        S1_VGG_in = self.resized_concat(I1) # Resize the input image, 
        S1_FEAS = self.vgg1(S1_VGG_in)
        
        S2_VGG_in = self.resized_concat(I2)
        S2_FEAS = self.vgg2(S2_VGG_in)

        generator_img = self.generator(I1.unsqueeze(1), I2.unsqueeze(1)) #64x64
        
        F_VGG_in = self.resized_concat(generator_img)
        
        F_FEAS = self.vggF(F_VGG_in)
        return S1_FEAS, S2_FEAS, F_FEAS, generator_img