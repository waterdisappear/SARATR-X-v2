import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from mmseg.models.losses.cross_entropy_loss import cross_entropy
from mmseg.models.losses.lovasz_loss import (
    flatten_probs,
    lovasz_softmax_flat,
)


def _lovasz_softmax_flat_safe(flat_probs, flat_labels, classes, class_weight):
    """避免 mmseg 在「无有效像素」时返回 [0,C] 张量，导致 per_image torch.stack 形状不一致。"""
    if flat_probs.numel() == 0:
        return flat_probs.sum()
    out = lovasz_softmax_flat(
        flat_probs,
        flat_labels,
        classes=classes,
        class_weight=class_weight)
    if out.numel() != 1:
        o = out.reshape(-1)
        if o.numel() == 0:
            return flat_probs.sum() * 0.0
        out = o.mean()
    return out


@LOSSES.register_module()
class CEAndLovaszLoss(nn.Module):
    """Weighted CE + Lovasz-Softmax for imbalanced segmentation."""

    def __init__(
            self,
            class_weight=None,
            ce_weight=1.0,
            lovasz_weight=0.5,
            per_image=True,
            classes='present',
            loss_weight=1.0,
            use_sigmoid=False,
            use_mask=False,
            reduction='mean',
            **kwargs):
        super().__init__()
        assert not use_sigmoid and not use_mask, (
            'CEAndLovaszLoss only supports softmax multi-class CE')
        self.class_weight = class_weight
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.per_image = per_image
        self.classes = classes
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        reduction = (
            reduction_override if reduction_override else self.reduction)
        cw = (cls_score.new_tensor(self.class_weight, dtype=torch.float32)
              if self.class_weight is not None else None)
        ce = cross_entropy(
            cls_score,
            label,
            weight=weight,
            class_weight=cw,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index)
        probs = F.softmax(cls_score, dim=1)
        if self.per_image:
            per_losses = []
            for i in range(probs.size(0)):
                fp, fl = flatten_probs(
                    probs[i:i + 1], label[i:i + 1], ignore_index)
                per_losses.append(
                    _lovasz_softmax_flat_safe(fp, fl, self.classes, cw))
            lov = torch.stack(per_losses).mean()
        else:
            fp, fl = flatten_probs(probs, label, ignore_index)
            lov = _lovasz_softmax_flat_safe(fp, fl, self.classes, cw)
        return self.loss_weight * (self.ce_weight * ce + self.lovasz_weight * lov)
