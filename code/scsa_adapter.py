
import torch
import torch.nn as nn
from scsa_attention import SCSA as SCSA_Original

class SCSA(nn.Module):

    def __init__(self, c1: int, c2: int, head_num: int=8):
        super().__init__()
        assert c1 == c2, 'Input and output channels must be equal for SCSA'
        self.sca = SCSA_Original(dim=c1, head_num=head_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sca(x)
