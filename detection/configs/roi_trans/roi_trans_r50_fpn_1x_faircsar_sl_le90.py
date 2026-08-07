# RoI Transformer (R50-FPN, le90) on FAIR-CSAR SL
# 检测头与 train/test 超参与官方 `roi_trans_r50_fpn_1x_dota_le90.py` 一致，仅替换数据与 SAR 归一化。
#
# 与论文表格严格对齐需自行核对：划分、epoch、batch/lr、测试增强等（见 FSI 同系列配置头注释）。
#
# 四卡全局 batch≈4：samples_per_gpu=1, lr=0.005；若 samples_per_gpu=2 则 lr≈0.01。
#
# 离线骨干：默认加载 ckpts/resnet50-0676ba61.pth 或 RESNET50_CKPT

import glob
import os

from mmrotate.datasets.dota import collect_dota_class_names_from_ann_folder

resnet50_ckpt = os.environ.get(
    'RESNET50_CKPT',
    'ckpts/resnet50-0676ba61.pth')

_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

angle_version = 'le90'

# 数据根目录：优先环境变量 FAIRCSAR_SL_TRAIN/TEST，否则使用默认相对路径 data/faircsar/
train_root = os.environ.get('FAIRCSAR_SL_TRAIN', 'data/faircsar/SL-TRAINVAL/trainval/')
test_root = os.environ.get('FAIRCSAR_SL_TEST', 'data/faircsar/SL-TRAINVAL/test/')

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
assert 'Tower_Crane' in CLASSES, (
    f"[FAIRCSAR] CLASSES 未包含 'Tower_Crane', num_classes={num_classes}")

assert os.path.isfile(resnet50_ckpt), (
    f'[offline] 未找到 ResNet50 权重: {resnet50_ckpt}\n'
    '默认应为 ckpts/resnet50-0676ba61.pth，或设置 RESNET50_CKPT')

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
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint=resnet50_ckpt)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RotatedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
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
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
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
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
