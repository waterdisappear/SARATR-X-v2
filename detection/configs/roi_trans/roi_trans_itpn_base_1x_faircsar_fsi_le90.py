# RoI Transformer + iTPN_pixel（base）on FAIR-CSAR FSI，1x（12 epoch）
#
# R50+RoI Trans 往往 12epoch 就够；iTPN 类骨干更吃训练长度，公平对比请优先用
# `roi_trans_itpn_base_3x_faircsar_fsi_le90.py`（36 epoch，与 ReDet+iTPN FSI 一致）。
# 本配置检测细节已与 `redet_itpn_base_3x_faircsar_fsi.py` 对齐（RPN、RiRoI、旋转增广）。
#
# 预训练权重：环境变量 ITPN_CKPT 优先，默认 mmrotate 安装目录 ckpts/iTPN/checkpoint-1200.pth

import glob
import os
import warnings

from mmrotate.datasets.dota import collect_dota_class_names_from_ann_folder

itpn_pretrained = os.environ.get(
    'ITPN_CKPT',
    'ckpts/iTPN/checkpoint-1200.pth')

_base_ = [
    '../_base_/schedules/schedule_1x.py',
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
        f"[FAIRCSAR-FSI] CLASSES 不包含 'Tower_Crane', num_classes={num_classes}",
        UserWarning)

assert os.path.isfile(itpn_pretrained), (
    f'[offline] 未找到 iTPN 权重: {itpn_pretrained}\n'
    '默认应为 ckpts/iTPN/checkpoint-1200.pth，或设置 ITPN_CKPT')

# 检测缩放（训练/测试输入）；与基线对齐
img_size = 1024
# iTPN_pixel 的 img_size 仅决定 __init__ 时 absolute_pos_embed 网格（官方 / checkpoint-1200 为 224→14×14）
# 若写成 1024 会导致预训练 pos_embed 形状不匹配而无法正确加载；前向里会对 pos 做插值适配 1024 输入
itpn_img_size = 224

img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
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
        img_scale=(1024, 1024),
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
    samples_per_gpu=1,
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
    type='RoITransformer',
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
        out_embed_dim=256,
        drop_path_rate=0.05,
        ape=True,
        rpe=False,
        patch_norm=True,
        use_checkpoint=False,
        num_classes=num_classes,
        init_cfg=dict(type='Pretrained', checkpoint=itpn_pretrained),
    ),
    neck=None,
    rpn_head=dict(
        type='RotatedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='RoITransRoIHead',
        version=angle_version,
        num_stages=2,
        stage_loss_weights=[1, 1],
        bbox_roi_extractor=[
            dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RiRoIAlignRotated',
                    out_size=7,
                    num_samples=2,
                    num_orientations=8,
                    clockwise=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
        ],
        bbox_head=[
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHAHBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=2,
                    edge_swap=True,
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=[0., 0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.5]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type=angle_version, iou_thr=0.1),
            max_per_img=2000)))

evaluation = dict(
    _delete_=True,
    interval=1,
    metric='mAP50',
    also_eval_on_test=True)

optimizer = dict(
    _delete_=True,
    # 参考 iTPN 官方 det（pixel_itpn_base_1x_ld090_dp005.py）：AdamW + layer-wise lr decay
    # 注：官方 COCO 配置用 lr=2e-4（典型大 batch）。本项目 FAIR-CSAR 常用全局 batch≈4，
    # 这里取更保守的 1.1e-4（与本仓库其它 iTPN 3x 配方一致），如增大总 batch 可线性放大 lr。
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
        }))

optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # 参考 iTPN 官方 det 1x：step=[9, 11]
    step=[9, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
