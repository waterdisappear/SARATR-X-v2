# ReDet + iTPN_pixel（base）on SIVED（车辆 SAR），3x（36 epoch）
# 参考 `configs/redet/redet_itpn_base_3x_rsar.py`：模型结构/优化器/调度；数据根目录默认 data/sived，可用 SIVED_ROOT 覆盖

import glob
import os
import warnings

from mmrotate.datasets.dota import collect_dota_class_names_from_ann_folder

_base_ = [
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py',
]

angle_version = 'le90'

# 数据根目录（默认 data/sived，可用 SIVED_ROOT 覆盖）
# 数据根目录：优先环境变量 SIVED_ROOT，否则使用默认相对路径 data/sived
data_root = os.environ.get('SIVED_ROOT', 'data/sived')

# SIVED 原始标注为 XML：`Annotations/*.xml`
# 但你已经有 DOTA 风格 TXT：`ImageSets/labelTxt/{train,val,test}/*.txt`（推荐直接用这个训练/评测）
#
# 期望目录结构（按你提供的信息）:
# <data_root>/
#   Annotations/*.xml               (不使用)
#   ImageSets/labelTxt/train/*.txt  (DOTA 标注，使用)
#   ImageSets/labelTxt/val/*.txt    (DOTA 标注，使用)
#   ImageSets/labelTxt/test/*.txt   (DOTA 标注，使用)
#   JPEGImages/* 或 PNGImages/*     (图片)
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
        f'[SIVED] 未检测到 val labelTxt（将跳过 val，仅评测 train 与 test）。'
        f' 若你确实有 val，请检查: {val_ann_dir}',
        UserWarning)

_cls_union = set()
for _d in _label_union_dirs:
    _cls_union |= set(collect_dota_class_names_from_ann_folder(_d, angle_version))
CLASSES = tuple(sorted(_cls_union))
num_classes = len(CLASSES)
assert num_classes > 0, f'[SIVED] 未从标注扫描到任何类别，请检查标注格式/类别列: {_label_union_dirs}'

# SAR 归一化（与 RSAR/FAIRCSAR 一致）
img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

# 检测缩放（输入分辨率）；SIVED 以 512 为主
img_size = 512
itpn_img_size = 224

train_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(type='LoadAnnotations', with_bbox=True),
    # 以 512 为中心的多尺度训练（square）
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
    type='ReDet',
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
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
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
            # 车辆目标可能细长，保留 RSAR 的更丰富比例
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
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
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
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
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
                debug=False),
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
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))

# AdamW + LayerDecay（参考 iTPN det / RSAR 配方）
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
    metric='mAP',
    iou_thr=[0.5, 0.75],
    also_eval_on_test=True)

checkpoint_config = dict(interval=1, max_keep_ckpts=4)

