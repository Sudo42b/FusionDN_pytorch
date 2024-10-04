import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHT_INIT_STDDEV = 0.05

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.target_features = None

    def forward(self, I1, I2):
        img = torch.cat([I1, I2], dim=1)
        self.target_features = self.encoder(img)
        generated_img = self.decoder(self.target_features)
        return generated_img


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 48, kernel_size=3, 
                               padding=1, stride=1, bias=True, 
                               padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(48)

        self.res_block1 = self._build_res_block(48, 48)
        self.res_block2 = self._build_res_block(96, 48)
        self.res_block3 = self._build_res_block(144, 48)
        self.res_block4 = self._build_res_block(192, 48)

        self._initialize_weights()  # Initialize weights with specified stddev

    def forward(self, image):
        out = F.relu(self.bn1(self.conv1(image)))
        out = torch.concat([out, self.res_block1(out)], dim=1)
        out = torch.concat([out, self.res_block2(out)], dim=1)
        out = torch.concat([out, self.res_block3(out)], dim=1)
        out = torch.concat([out, self.res_block4(out)], dim=1)
        return out

    def _build_res_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, 
                    stride=1, bias=True, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, 
                    stride=1, bias=True, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def _initialize_weights(self):
        # Initialize the first conv layer with WEIGHT_INIT_STDDEV
        nn.init.normal_(self.conv1.weight, mean=0.0, std=WEIGHT_INIT_STDDEV)
        nn.init.zeros_(self.conv1.bias)

        # Initialize other conv layers with sqrt(2 / ch)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m != self.conv1:
                stddev = (2 / m.out_channels) ** 0.5
                nn.init.normal_(m.weight, mean=0.0, std=stddev)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.05)
                nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv2_1 = nn.Conv2d(240, 240, kernel_size=3, padding=1, 
                                 stride=1, bias=True, padding_mode='reflect')
        
        self.conv2_2 = nn.Conv2d(240, 128, kernel_size=3, padding=1, 
                                 stride=1, bias=True, padding_mode='reflect')
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, 
                                 stride=1, bias=True, padding_mode='reflect')
        self.bn2_3 = nn.BatchNorm2d(64)
        
        self.conv2_4 = nn.Conv2d(64, 1, kernel_size=3, padding=1, 
                                 stride=1, bias=True, padding_mode='reflect')

        self._initialize_weights()  # Initialize weights with specified stddev

    def forward(self, x):
        out = F.relu(self.conv2_1(x))
        out = F.relu(self.bn2_2(self.conv2_2(out)))
        out = F.relu(self.bn2_3(self.conv2_3(out)))
        out = torch.tanh(self.conv2_4(out)) / 2 + 0.5
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=WEIGHT_INIT_STDDEV)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.05)
                nn.init.constant_(m.bias, 0)
