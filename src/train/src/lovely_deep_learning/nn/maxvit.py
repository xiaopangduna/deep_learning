import torch
import torch.nn as nn
from torchvision.models import maxvit_t


class MaxVitT(nn.Module):
    """
    YAML-friendly MaxVit-T.

    We materialize torchvision's maxvit_t submodules directly (stem/blocks/classifier)
    so state_dict keys stay compatible with torchvision:
    - stem.*
    - blocks.*
    - classifier.*
    """

    def __init__(self):
        super().__init__()
        model = maxvit_t(weights=None)
        self.stem = model.stem
        self.blocks = model.blocks
        self.classifier = model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

