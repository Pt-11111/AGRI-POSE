

import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv

class PoseDFL(Detect):
    KPT_RANGE = 20
    N_BINS = 24
    REG_RANGE = 1.0 / (N_BINS - 1)

    def __init__(self, nc: int=80, kpt_shape: tuple=(17, 3), ch: tuple=()):
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape
        self.nk = kpt_shape[0] * kpt_shape[1]
        self.nkpt = kpt_shape[0]
        self.nk_dfl_loc = self.nkpt * 2 * self.N_BINS
        self.nk_vis = self.nkpt * 1
        self.nk_total = self.nk_dfl_loc + self.nk_vis
        c4 = max(ch[0] // 4, self.nk_total)
        self.cv4 = nn.ModuleList((nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk_total, 1)) for x in ch))
        self.bins = torch.linspace(0, self.N_BINS - 1, self.N_BINS) * self.REG_RANGE
        self.register_buffer('bins_tensor', self.bins)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        bs = x[0].shape[0]
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk_total, -1) for i in range(self.nl)], -1)
        x = Detect.forward(self, x)
        if self.training:
            return (x, kpt)
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs: int, kpts: torch.Tensor) -> torch.Tensor:
        ndim = self.kpt_shape[1]
        num_anchors = kpts.shape[-1]
        kpt_num = self.kpt_shape[0]
        loc_distri = kpts[:, :self.nk_dfl_loc, :]
        loc_distri = loc_distri.permute(0, 2, 1).contiguous().view(bs, num_anchors, kpt_num, 2, self.N_BINS)
        prob = loc_distri.softmax(dim=-1)
        pred_rel = torch.sum(prob * self.bins_tensor, dim=-1)
        vis_logits = kpts[:, self.nk_dfl_loc:, :].permute(0, 2, 1).contiguous().unsqueeze(-1)
        y = torch.cat([pred_rel, vis_logits], dim=-1)
        if self.export:
            pass
        else:
            if ndim == 3:
                y[..., 2] = y[..., 2].sigmoid()
            anchor_x = self.anchors[0]
            anchor_y = self.anchors[1]
            stride = self.strides
            offset = self.KPT_RANGE / 2.0
            y[..., 0] = (y[..., 0] * self.KPT_RANGE + (anchor_x.view(1, -1, 1) - offset)) * stride.view(1, -1, 1)
            y[..., 1] = (y[..., 1] * self.KPT_RANGE + (anchor_y.view(1, -1, 1) - offset)) * stride.view(1, -1, 1)
            return y.view(bs, num_anchors, -1).permute(0, 2, 1).contiguous()
