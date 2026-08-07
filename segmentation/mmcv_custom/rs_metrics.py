# -*- coding: utf-8 -*-
"""Remote-sensing style metrics on top of mmseg-style label handling."""

import os.path as osp
import numpy as np

try:
    import mmcv
except ImportError:
    mmcv = None


def _align_pred_to_label(pred, label):
    pred = np.asarray(pred)
    label = np.asarray(label)
    if pred.shape != label.shape:
        if mmcv is None:
            raise ImportError('mmcv is required when pred and label shapes differ')
        pred = mmcv.imresize(
            pred, (label.shape[1], label.shape[0]), interpolation='nearest')
    return pred, label


def _prepare_label_like_mmseg(label, reduce_zero_label, ignore_index):
    """Match mmseg ``intersect_and_union`` label handling (numpy)."""
    label = np.asarray(label)
    if reduce_zero_label:
        label_copy = label.copy()
        label[label_copy == 0] = ignore_index
        label = label - 1
        label[label == 254] = ignore_index
    return label


def _get_gt_seg_map_by_idx(dataset, idx):
    """与官方 CustomDataset 读盘方式一致；无 ``get_gt_seg_map_by_idx`` 时回退单张读取。"""
    if hasattr(dataset, 'get_gt_seg_map_by_idx'):
        return np.asarray(dataset.get_gt_seg_map_by_idx(idx))
    if mmcv is None:
        raise ImportError('mmcv is required to read GT for confusion matrix')
    ann = dataset.img_infos[idx]['ann']['seg_map']
    p = osp.join(dataset.ann_dir, ann)
    return mmcv.imread(p, flag='unchanged', backend='pillow')


def accumulate_confusion_matrix(
        preds,
        dataset,
        num_classes,
        ignore_index=255,
        reduce_zero_label=False,
        label_map=None):
    """Build full (num_classes, num_classes) confusion: rows=gt, cols=pred."""
    if label_map is None:
        label_map = dict()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n = len(preds)
    for i in range(n):
        pred = np.asarray(preds[i])
        gt = np.asarray(_get_gt_seg_map_by_idx(dataset, i))
        pred, gt = _align_pred_to_label(pred, gt)
        if label_map:
            gt_copy = gt.copy()
            for old_id, new_id in label_map.items():
                gt[gt_copy == old_id] = new_id
        gt = _prepare_label_like_mmseg(gt, reduce_zero_label, ignore_index)
        mask = gt != ignore_index
        pred = pred[mask]
        gt = gt[mask]
        pred = np.clip(pred, 0, num_classes - 1)
        gt = np.clip(gt, 0, num_classes - 1)
        idx = gt.astype(np.int64) * num_classes + pred.astype(np.int64)
        cm += np.bincount(idx, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes)
    return cm


def overall_accuracy_from_confusion(cm):
    n = cm.sum()
    if n == 0:
        return 0.0
    return float(np.trace(cm) / n)


def kappa_from_confusion(cm):
    """Multiclass Kappa (same as sklearn ``cohen_kappa_score`` with linear weights)."""
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    row = cm.sum(axis=1).astype(np.float64)
    col = cm.sum(axis=0).astype(np.float64)
    pe = np.dot(row, col) / (n * n)
    denom = 1.0 - pe
    if abs(denom) < 1e-12:
        return 0.0
    return float((po - pe) / denom)


def is_pre_eval_format(results):
    if not results:
        return False
    r0 = results[0]
    if not isinstance(r0, (tuple, list)):
        return False
    return len(r0) == 4
