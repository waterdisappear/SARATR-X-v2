# OrientedRCNN + iTPN_pixel on FAIR-CSAR (SL-TRAINVAL / SL-TEST)
# 以 `oriented_rcnn_itpn_base_3x_rsar.py` 为模板改写

import glob
import os

from mmrotate.datasets.dota import collect_dota_class_names_from_ann_folder

_base_ = [
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py',
]

angle_version = 'le90'

# ======================
# 路径参数（需要你自己改这里）
# ======================
# 数据根目录：优先环境变量 FAIRCSAR_SL_TRAIN/TEST，否则使用默认相对路径 data/faircsar/
train_root = os.environ.get('FAIRCSAR_SL_TRAIN', 'data/faircsar/SL-TRAINVAL/trainval/')
test_root = os.environ.get('FAIRCSAR_SL_TEST', 'data/faircsar/SL-TRAINVAL/test/')

train_ann_dir = os.path.join(train_root, 'Annotations')
train_img_dir = os.path.join(train_root, 'PNGImages')
test_ann_dir = os.path.join(test_root, 'Annotations')
test_img_dir = os.path.join(test_root, 'PNGImages')

# iTPN 官方 det 预训练权重（pixel itpn base）
pretrained = 'ckpts/iTPN/checkpoint-1200.pth'

# 若 import 时路径不对或 train 下没有 txt，CLASSES 会缺类，训练阶段才报 warning。
assert os.path.isdir(train_ann_dir), (
    f'[FAIRCSAR] train_ann_dir 不存在: {train_ann_dir} '
    '(请改 config 顶部 train_root / train_ann_dir)')
assert os.path.isdir(test_ann_dir), (
    f'[FAIRCSAR] test_ann_dir 不存在: {test_ann_dir} '
    '(请改 config 顶部 test_root / test_ann_dir)')
_n_train_txt = len(glob.glob(os.path.join(train_ann_dir, '*.txt')))
_n_test_txt = len(glob.glob(os.path.join(test_ann_dir, '*.txt')))
assert _n_train_txt > 0, (
    f'[FAIRCSAR] train_ann_dir 下没有 *.txt: {train_ann_dir} '
    f'(当前 glob 到 {_n_train_txt} 个文件)')
assert _n_test_txt > 0, (
    f'[FAIRCSAR] test_ann_dir 下没有 *.txt: {test_ann_dir} '
    f'(当前 glob 到 {_n_test_txt} 个文件)')

# 类别名：与 DOTADataset.load_annotations 完全一致（须与下方 data.*.label_union_dirs 并集一致）
_cls_train = set(collect_dota_class_names_from_ann_folder(train_ann_dir, angle_version))
_cls_test = set(collect_dota_class_names_from_ann_folder(test_ann_dir, angle_version))
CLASSES = tuple(sorted(_cls_train | _cls_test))
num_classes = len(CLASSES)
_label_union_dirs = [train_ann_dir, test_ann_dir]
assert 'Tower_Crane' in CLASSES, (
    f"[FAIRCSAR] scan CLASSES failed to include 'Tower_Crane'. "
    f"num_classes={num_classes}. "
    f"CLASSES preview={list(CLASSES)[:30]}")


img_size = 1024
# iTPN_pixel 的 img_size 仅决定 __init__ 时 absolute_pos_embed 网格（官方 / checkpoint-1200 为 224→14×14）
# 若写成 1024 会导致预训练 pos_embed 形状不匹配而无法正确加载；前向里会对 pos 做插值适配 1024 输入
itpn_img_size = 224
train_img_scales = [(704, 704), (832, 832), (960, 960), (1024, 1024)]

# 数据增强/归一化（保持与 RSAR 配置一致；如需可继续调）
img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RResize',
        img_scale=train_img_scales,
        multiscale_mode='value'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    # 随机多边形旋转增强，利于旋转目标鲁棒性
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
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
    # 有效 batch ≈ samples_per_gpu * num_gpus；OOM 时可改为 1
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        # 类均衡重采样：增强小样本类（如 Tower_Crane / Other_Aircraft）
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type='FAIRCSARDataset',
            ann_file=train_ann_dir,
            img_prefix=train_img_dir,
            pipeline=train_pipeline,
            version=angle_version,
            label_union_dirs=_label_union_dirs,
            split='train',
            val_ratio=0.1,
            seed=42)),
    # 从 trainval/Annotations 中随机切 10% 作为验证集
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
    type='OrientedRCNN',
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
        # 以算力换显存：主 24 层 block 用 gradient checkpoint
        use_checkpoint=True,
        num_classes=num_classes,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=None,
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            # 针对细长目标增加更多长宽比
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
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
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
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
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            # 更激进一点的 recall：略降阈值、保留更多候选
            score_thr=0.02,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))


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

