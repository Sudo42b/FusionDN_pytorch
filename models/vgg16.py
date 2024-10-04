"""
    Reference:
        https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html

    VGG16 model architecture from the "Very Deep Convolutional Networks for Large-Scale Image Recognition" paper.
    The weights of this model are initialized by the weights from the ImageNet classification task.
    The model is from the `torchvision.models` module.
"""

from torchvision.models import VGG16_Weights 
# Importing the VGG16_Weights class from torchvision.models
# This class is used to specify the weights of the VGG16 model.
# The `VGG_MEANO' of [103.939, 116.779, 123.68] is used for normalization.
# RGB to BGR conversion is done for the input images.
# Division by 255 is done for the input images.
# ImageNet mean=(0.485, 0.456, 0.406) is used for normalization. (This is the mean of the ImageNet dataset)

from torchvision.models._api import WeightsEnum
from torchvision.models.vgg import make_layers
from torchvision.models._utils import _ovewrite_named_param
from torchvision.utils import _log_api_usage_once
from torch import nn
from typing import List, Union, Dict, Any, Optional, cast
import torch

cfgs: Dict[str, List[Union[str, int]]] = {
    # VGG11
    # "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], 
    # VGG13
    # "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], 
    # VGG16
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], 
    # VGG19
    # "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"], 
}

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        output = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in [3, 8, 15, 22, 29]:
                output.append(x)
        return output


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(cfg: str, batch_norm: bool, weights: Optional[WeightsEnum], progress: bool, **kwargs: Any) -> VGG:
    if weights is not None:
        kwargs["init_weights"] = False
        if weights.meta["categories"] is not None:
            _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
    return model


def vgg16(*, weights: Optional[VGG16_Weights] = None, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image Recognition <https://arxiv.org/abs/1409.1556>`__.

    Args:
        weights (:class:`~torchvision.models.VGG16_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.VGG16_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vgg.VGG``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.VGG16_Weights
        :members:
    """
    weights = VGG16_Weights.verify(weights)

    return _vgg("D", False, weights, progress, **kwargs)


def get_vgg16(weights: Optional[VGG16_Weights] = 'IMAGENET1K_FEATURES', 
              progress: bool = True, **kwargs: Any) -> VGG:
    return vgg16(weights=weights, progress=progress, **kwargs)