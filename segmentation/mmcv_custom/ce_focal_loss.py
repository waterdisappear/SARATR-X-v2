import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES


@LOSSES.register_module()
class CEFocalLoss(nn.Module):
    """Combined Cross-Entropy and Focal Loss for segmentation."""

    def __init__(self,
                 use_sigmoid=False,
                 gamma=2.0,
                 alpha=0.25,
                 class_weight=None,
                 ce_weight=1.0,
                 focal_weight=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super().__init__()
        if use_sigmoid:
            raise NotImplementedError('CEFocalLoss currently supports softmax only.')
        self.gamma = gamma
        self.alpha = alpha
        self.class_weight = class_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255):
        reduction = reduction_override if reduction_override else self.reduction

        cls_weight = None
        if self.class_weight is not None:
            cls_weight = pred.new_tensor(self.class_weight, dtype=torch.float32)

        ce_loss = F.cross_entropy(
            pred,
            target,
            weight=cls_weight,
            reduction='none',
            ignore_index=ignore_index)

        valid_mask = (target != ignore_index).float()
        pt = torch.exp(-ce_loss)
        focal = self.alpha * torch.pow((1.0 - pt).clamp(min=0.0), self.gamma) * ce_loss
        loss = self.ce_weight * ce_loss + self.focal_weight * focal
        loss = loss * valid_mask

        if weight is not None:
            if weight.dim() == 4 and weight.size(1) == 1:
                weight = weight.squeeze(1)
            loss = loss * weight
            valid_mask = valid_mask * weight

        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            if avg_factor is None:
                denom = valid_mask.sum().clamp(min=1.0)
            else:
                denom = pred.new_tensor(float(avg_factor)).clamp(min=1.0)
            loss = loss.sum() / denom
        elif reduction != 'none':
            raise ValueError(f'Unsupported reduction: {reduction}')

        return self.loss_weight * loss
