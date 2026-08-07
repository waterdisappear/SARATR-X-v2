# 固定训练尺度 512 的消融版本（仅改数据增强尺度策略）
_base_ = ['./rotated_faster_rcnn_itpn_base_3x_sived_le90.py']

angle_version = 'le90'

img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(512, 512)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    train=dict(pipeline=train_pipeline, version=angle_version),
)

