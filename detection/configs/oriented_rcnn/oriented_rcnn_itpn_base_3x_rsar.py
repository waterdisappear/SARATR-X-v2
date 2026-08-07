_base_ = [
    '../_base_/datasets/rsar.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le90'

# iTPN 官方 det 预训练权重（pixel itpn base），需放置到 ckpts/iTPN/
pretrained = 'ckpts/iTPN/checkpoint-1200.pth'
# iTPN_pixel 的 img_size 仅决定 __init__ 时 absolute_pos_embed 网格（官方 / checkpoint-1200 为 224→14×14）
# 检测实际输入分辨率由数据 pipeline 决定（如 800/1024）；前向会对 pos 做插值适配
itpn_img_size = 224

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
        use_checkpoint=False,
        num_classes=6,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    # iTPN_pixel 已自带 FPN 风格多尺度特征，这里不再额外接 FPN
    neck=None,
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            # 针对 RSAR 中细长目标，增加更扁/更细长的比例
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
            num_classes=6,
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

# 优化器：沿用 iTPN+Cosine 的风格，略微增大学习率并延长训练轮次
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=8e-5,
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
            'backbone': dict(lr_mult=0.2),  # 稍提高骨干学习率，更好适应 RSAR
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

# 与 ReDet 版保持一致：3x 36 epoch
runner = dict(type='EpochBasedRunner', max_epochs=36)

evaluation = dict(
    interval=1,
    metric='mAP50',
    also_eval_on_test=True)

checkpoint_config = dict(interval=1, max_keep_ckpts=4)

# 加强数据增强：加入随机旋转；并显式写全 data，避免覆盖 base 后缺少 val/test 导致 cfg.data 报错
_data_root = 'data/rsar/'
_img_norm = dict(mean=[53.7795, 53.7795, 53.7795], std=[55.539, 55.539, 55.539], to_rgb=True)
_test_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(type='MultiScaleFlipAug', img_scale=(800, 800), flip=False, transforms=[
        dict(type='RResize'),
        dict(type='Normalize', **_img_norm),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img']),
    ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RSARDataset',
        ann_file=_data_root + 'train/annfiles/',
        img_prefix=_data_root + 'train/images/',
        pipeline=[
            dict(type='LoadImageFromFileMultiExt'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=[(640, 640), (800, 800), (960, 960)], multiscale_mode='value'),
            dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25], direction=['horizontal', 'vertical', 'diagonal'], version='le90'),
            dict(type='PolyRandomRotate', rotate_ratio=0.5, mode='range', angles_range=180, auto_bound=False, version='le90'),
            dict(type='Normalize', **_img_norm),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ],
        version='le90',
    ),
    val=dict(
        type='RSARDataset',
        ann_file=_data_root + 'val/annfiles/',
        img_prefix=_data_root + 'val/images/',
        pipeline=_test_pipeline,
        version='le90',
    ),
    test=dict(
        type='RSARDataset',
        ann_file=_data_root + 'test/annfiles/',
        img_prefix=_data_root + 'test/images/',
        pipeline=_test_pipeline,
        version='le90',
    ),
)

