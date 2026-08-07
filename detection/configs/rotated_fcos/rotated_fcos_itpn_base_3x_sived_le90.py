# Rotated FCOS (anchor-free one-stage) + iTPN_pixel（base）on SIVED，3x（36 epoch）
# 超参参考：configs/redet/redet_itpn_base_3x_rsar.py（AdamW+LayerDecay+Cosine）

import glob
import os
import warnings

from mmrotate.datasets.dota import collect_dota_class_names_from_ann_folder

_base_ = [
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py',
]

angle_version = 'le90'

# 数据根目录：优先环境变量 SIVED_ROOT，否则使用默认相对路径 data/sived
data_root = os.environ.get('SIVED_ROOT', 'data/sived')
train_ann_dir = os.path.join(data_root, 'ImageSets', 'labelTxt', 'train')
val_ann_dir = os.path.join(data_root, 'ImageSets', 'labelTxt', 'val')
test_ann_dir = os.path.join(data_root, 'ImageSets', 'labelTxt', 'test')

_img_dir_candidates = [
    os.path.join(data_root, 'JPEGImages'),
    os.path.join(data_root, 'PNGImages'),
    os.path.join(data_root, 'Images'),
]
img_dir = next((d for d in _img_dir_candidates if os.path.isdir(d)), None)

pretrained = 'ckpts/iTPN/checkpoint-1200.pth'
itpn_img_size = 224

assert os.path.isdir(train_ann_dir), f'[SIVED] train labelTxt 不存在: {train_ann_dir}'
assert os.path.isdir(test_ann_dir), f'[SIVED] test labelTxt 不存在: {test_ann_dir}'
assert len(glob.glob(os.path.join(train_ann_dir, '*.txt'))) > 0, f'[SIVED] {train_ann_dir} 下没有 *.txt'
assert len(glob.glob(os.path.join(test_ann_dir, '*.txt'))) > 0, f'[SIVED] {test_ann_dir} 下没有 *.txt'
assert img_dir is not None and os.path.isdir(img_dir), (
    f'[SIVED] 未找到图片目录，请确认存在 JPEGImages/PNGImages/Images 之一。'
    f' tried={_img_dir_candidates}')

_label_union_dirs = [train_ann_dir, test_ann_dir]
if os.path.isdir(val_ann_dir) and len(glob.glob(os.path.join(val_ann_dir, '*.txt'))) > 0:
    _label_union_dirs.insert(1, val_ann_dir)
else:
    warnings.warn(
        f'[SIVED] 未检测到 val labelTxt（val 将回退 train）。路径: {val_ann_dir}',
        UserWarning)

_cls_union = set()
for _d in _label_union_dirs:
    _cls_union |= set(collect_dota_class_names_from_ann_folder(_d, angle_version))
CLASSES = tuple(sorted(_cls_union))
num_classes = len(CLASSES)
assert num_classes > 0, f'[SIVED] 未扫描到类别: {_label_union_dirs}'

img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

img_size = 512

train_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RResize',
        img_scale=[(448, 448), (512, 512), (576, 576)],
        multiscale_mode='value'),
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

test_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FAIRCSARDataset',
        ann_file=train_ann_dir,
        img_prefix=img_dir,
        pipeline=train_pipeline,
        version=angle_version,
        classes=CLASSES,
        label_union_dirs=_label_union_dirs),
    val=dict(
        type='FAIRCSARDataset',
        ann_file=val_ann_dir if os.path.isdir(val_ann_dir) else train_ann_dir,
        img_prefix=img_dir,
        pipeline=test_pipeline,
        version=angle_version,
        classes=CLASSES,
        label_union_dirs=_label_union_dirs),
    test=dict(
        type='FAIRCSARDataset',
        ann_file=test_ann_dir,
        img_prefix=img_dir,
        pipeline=test_pipeline,
        version=angle_version,
        classes=CLASSES,
        label_union_dirs=_label_union_dirs),
)

model = dict(
    type='RotatedFCOS',
    backbone=dict(
        _delete_=True,
        type='iTPN_pixel',
        img_size=itpn_img_size,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        mlp_depth=3,
        fpn_dim=256,
        fpn_depth=1,
        depth=24,
        num_heads=8,
        bridge_mlp_ratio=3.,
        mlp_ratio=4.,
        num_outs=5,
        drop_path_rate=0.05,
        ape=True,
        rpe=False,
        patch_norm=True,
        use_checkpoint=False,
        num_classes=num_classes,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=None,
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000),
)

# AdamW + LayerDecay（参考 RSAR iTPN 配方）
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=30,
        layer_decay_rate=0.90,
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.2),
        }))

optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    min_lr_ratio=1e-6,
    by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=36)

evaluation = dict(
    interval=1,
    metric='mAP',
    iou_thr=[0.5, 0.75],
    also_eval_on_test=True)

checkpoint_config = dict(interval=1, max_keep_ckpts=4)

