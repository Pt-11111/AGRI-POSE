
from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss, KeypointLoss
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.tal import make_anchors

class v8PoseDFL(v8DetectionLoss):

    def __init__(self, model):
        super().__init__(model)
        self.pose_head = pose_head = model.model[-1]
        self.kpt_shape = pose_head.kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [2, 3]
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
        self.N_BINS = pose_head.N_BINS
        self.REG_RANGE = pose_head.REG_RANGE
        self.KPT_RANGE = pose_head.KPT_RANGE
        self.bins_tensor = pose_head.bins_tensor
        self.nkpt = pose_head.kpt_shape[0]
        self.nk_dfl_loc = pose_head.nk_dfl_loc
        self.dfl_ce_loss = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        loss = torch.zeros(6, device=self.device)
        (feats, pred_kpts_raw) = preds if isinstance(preds[0], list) else preds[1]
        (pred_distri, pred_scores) = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        (anchor_points, stride_tensor) = make_anchors(feats, self.stride, 0.5)
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        (gt_labels, gt_bboxes) = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts_decoded = self.kpts_decode(pred_kpts_raw, anchor_points)
        pred_kpts_oks = pred_kpts_decoded
        (_, target_bboxes, target_scores, fg_mask, target_gt_idx) = self.assigner(pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
        target_scores_sum = max(target_scores.sum(), 1)
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            (loss[0], loss[4]) = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            (kpts_oks_loss, kpts_obj_loss, kpts_dfl_loss) = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts_oks, pred_kpts_raw, anchor_points)
            loss[1] = kpts_oks_loss
            loss[2] = kpts_obj_loss
            loss[5] = kpts_dfl_loss
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl
        loss[5] *= self.hyp.pose_dfl
        return (loss.sum() * batch_size, loss.detach())

    def kpts_decode(self, pred_kpts, anchor_points):
        (bs, _, n_anchors) = pred_kpts.shape
        kpt_num = self.kpt_shape[0]
        loc_logits = pred_kpts[:, :self.nk_dfl_loc, :]
        vis_logits = pred_kpts[:, self.nk_dfl_loc:, :]
        loc_logits = loc_logits.permute(0, 2, 1).contiguous().view(bs, n_anchors, kpt_num, 2, self.N_BINS)
        prob = loc_logits.softmax(dim=-1)
        xy_rel = (prob * self.bins_tensor).sum(dim=-1)
        offset = self.KPT_RANGE / 2.0
        xy_grid = xy_rel * self.KPT_RANGE
        xy_grid[..., 0] += anchor_points[:, 0].view(1, -1, 1) - offset
        xy_grid[..., 1] += anchor_points[:, 1].view(1, -1, 1) - offset
        vis = vis_logits.permute(0, 2, 1).contiguous().unsqueeze(-1)
        return torch.cat([xy_grid, vis], dim=-1)

    def kpts_target_construct(self, gt_kpt_rel: torch.Tensor):
        gt_kpt_rel = gt_kpt_rel.clamp(0.0, self.N_BINS * self.REG_RANGE - 1e-06)
        x = gt_kpt_rel[..., 0]
        y = gt_kpt_rel[..., 1]
        l_x_idx = (x / self.REG_RANGE).long()
        r_x_idx = l_x_idx + 1
        l_y_idx = (y / self.REG_RANGE).long()
        r_y_idx = l_y_idx + 1
        weight_r_x = x / self.REG_RANGE - l_x_idx.float()
        weight_l_x = 1.0 - weight_r_x
        weight_r_y = y / self.REG_RANGE - l_y_idx.float()
        weight_l_y = 1.0 - weight_r_y
        l_x_idx = l_x_idx.clamp(min=0, max=self.N_BINS - 1)
        r_x_idx = r_x_idx.clamp(min=0, max=self.N_BINS - 1)
        l_y_idx = l_y_idx.clamp(min=0, max=self.N_BINS - 1)
        r_y_idx = r_y_idx.clamp(min=0, max=self.N_BINS - 1)
        N = gt_kpt_rel.shape[0]
        target_dist = torch.zeros((N, 2 * self.N_BINS), device=gt_kpt_rel.device)
        row = torch.arange(N, device=gt_kpt_rel.device)
        target_dist[row, l_x_idx] = weight_l_x
        target_dist[row, r_x_idx] = weight_r_x
        target_dist[row, l_y_idx + self.N_BINS] = weight_l_y
        target_dist[row, r_y_idx + self.N_BINS] = weight_r_y
        return target_dist

    def calculate_keypoints_loss(self, masks: torch.Tensor, target_gt_idx: torch.Tensor, keypoints: torch.Tensor, batch_idx: torch.Tensor, stride_tensor: torch.Tensor, target_bboxes: torch.Tensor, pred_kpts: torch.Tensor, pred_kpts_raw: torch.Tensor, anchor_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
        batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device)
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_keypoints.gather(1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        kpts_loss = 0
        kpts_obj_loss = 0
        kpts_dfl_loss = 0
        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
            fg_kpt_raw = pred_kpts_raw.permute(0, 2, 1).contiguous()[masks]
            dfl_logits = fg_kpt_raw[..., :self.nk_dfl_loc]
            dfl_logits_reshaped = dfl_logits.reshape(-1, 2 * self.N_BINS)
            mask_flat = kpt_mask.flatten()
            if mask_flat.sum() > 0:
                dfl_logits_valid = dfl_logits_reshaped[mask_flat]
                anchor_points_batched = anchor_points.unsqueeze(0).repeat(batch_size, 1, 1)
                anchor_points_fg = anchor_points_batched[masks]
                anchor_points_expanded = anchor_points_fg.unsqueeze(1).expand(-1, self.kpt_shape[0], 2)
                anchor_valid = anchor_points_expanded.reshape(-1, 2)[mask_flat]
                gt_kpt_loc_valid = gt_kpt[..., :2].reshape(-1, 2)[mask_flat]
                target_dist_rel = (gt_kpt_loc_valid - (anchor_valid - self.KPT_RANGE / 2.0)) / self.KPT_RANGE
                if target_dist_rel.max().item() > 2.0:
                    print('[WARNING] Target value is suspiciously large! Check stride division.')
                target_dist = self.kpts_target_construct(target_dist_rel)
                pred_x_dist = dfl_logits_valid[:, :self.N_BINS]
                pred_y_dist = dfl_logits_valid[:, self.N_BINS:]
                target_x_dist = target_dist[:, :self.N_BINS]
                target_y_dist = target_dist[:, self.N_BINS:]
                dfl_x_loss = -(target_x_dist * F.log_softmax(pred_x_dist, dim=1)).sum(dim=1)
                dfl_y_loss = -(target_y_dist * F.log_softmax(pred_y_dist, dim=1)).sum(dim=1)
                kpts_dfl_loss = (dfl_x_loss.sum() + dfl_y_loss.sum()) / (dfl_logits_valid.shape[0] / 2 + 1e-09)
        return (kpts_loss, kpts_obj_loss, kpts_dfl_loss)
