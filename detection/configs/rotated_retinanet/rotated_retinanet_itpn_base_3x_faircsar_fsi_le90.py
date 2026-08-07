# RotatedRetinaNet（单阶段）+ iTPN_pixel，FAIR-CSAR FSI
# 数据 / 优化器 / 调度与 `redet_itpn_base_3x_faircsar_fsi.py` 对齐；neck=None，五层特征与 ReDet 的 RPN stride 一致

import glob
import os
import warnings

from mmrotate.datasets.dota import collect_dota_class_names_from_ann_folder

_base_ = [
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py',
]

angle_version = 'le90'

# 数据根目录：优先环境变量 FAIRCSAR_FSI_TRAIN/TEST，否则使用默认相对路径 data/faircsar/
train_root = os.environ.get('FAIRCSAR_FSI_TRAIN', 'data/faircsar/FSI-TRAINVAL/trainval/')
test_root = os.environ.get('FAIRCSAR_FSI_TEST', 'data/faircsar/FSI-TRAINVAL/test/')

train_ann_dir = os.path.join(train_root, 'Annotations')
train_img_dir = os.path.join(train_root, 'PNGImages')
test_ann_dir = os.path.join(test_root, 'Annotations')
test_img_dir = os.path.join(test_root, 'PNGImages')

pretrained = 'ckpts/iTPN/checkpoint-1200.pth'

assert os.path.isdir(train_ann_dir), (
    f'[FAIRCSAR] train_ann_dir 不存在: {train_ann_dir}')
assert os.path.isdir(test_ann_dir), (
    f'[FAIRCSAR] test_ann_dir 不存在: {test_ann_dir}')
assert len(glob.glob(os.path.join(train_ann_dir, '*.txt'))) > 0, (
    f'[FAIRCSAR] train_ann_dir 下没有 *.txt: {train_ann_dir}')
assert len(glob.glob(os.path.join(test_ann_dir, '*.txt'))) > 0, (
    f'[FAIRCSAR] test_ann_dir 下没有 *.txt: {test_ann_dir}')

_cls_train = set(collect_dota_class_names_from_ann_folder(train_ann_dir, angle_version))
_cls_test = set(collect_dota_class_names_from_ann_folder(test_ann_dir, angle_version))
CLASSES = tuple(sorted(_cls_train | _cls_test))
num_classes = len(CLASSES)
_label_union_dirs = [train_ann_dir, test_ann_dir]
if 'Tower_Crane' not in CLASSES:
    warnings.warn(
        f"[FAIRCSAR-FSI] 扫描到的 CLASSES 不包含 'Tower_Crane'，"
        f"num_classes={num_classes}. CLASSES preview={CLASSES[:10]}",
        UserWarning)

img_size = 1024
train_img_scale = (img_size, img_size)

img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RResize',
        img_scale=train_img_scale),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.3, 0.3, 0.3],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.0,
        mode='range',
        angles_range=180,
        auto_bound=False,
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
    workers_per_gpu=4,
    train=dict(
        type='FAIRCSARDataset',
        ann_file=train_ann_dir,
        img_prefix=train_img_dir,
        pipeline=train_pipeline,
        version=angle_version,
        label_union_dirs=_label_union_dirs,
        split='train',
        val_ratio=0.1,
        seed=42),
    val=dict(
        type='FAIRCSARDataset',
        ann_file=train_ann_dir,
        img_prefix=train_img_dir,
        pipeline=test_pipeline,
        version=angle_version,
        label_union_dirs=_label_union_dirs,
        split='val',
        val_ratio=0.1,
        seed=42),
    test=dict(
        type='FAIRCSARDataset',
        ann_file=test_ann_dir,
        img_prefix=test_img_dir,
        pipeline=test_pipeline,
        version=angle_version,
        label_union_dirs=_label_union_dirs),
)

model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        _delete_=True,
        type='iTPN_pixel',
        img_size=224,
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
        out_embed_dim=256,
        drop_path_rate=0.05,
        ape=True,
        rpe=False,
        patch_norm=True,
        use_checkpoint=True,
        num_classes=num_classes,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=None,
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1.1e-4,
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
    warmup_iters=1500,
    warmup_ratio=0.001,
    min_lr_ratio=5e-3,
    by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=36)

evaluation = dict(
    interval=1,
    metric='mAP50',
    also_eval_on_test=True)

checkpoint_config = dict(interval=1, max_keep_ckpts=4)

custom_hooks = [
    dict(
        type='SWAHook',
        swa_start=24,
        eval_with_swa=True,
        save_at_end=True,
        priority=40),
    dict(type='SWARestoreHook', priority=55),
]
