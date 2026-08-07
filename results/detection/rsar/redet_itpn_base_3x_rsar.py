dataset_type = 'RSARDataset'
data_root = 'data/rsar/'
img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RResize',
        img_scale=[(640, 640), (800, 800), (960, 960)],
        multiscale_mode='value'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(
        type='Normalize',
        mean=[53.7795, 53.7795, 53.7795],
        std=[55.539, 55.539, 55.539],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[53.7795, 53.7795, 53.7795],
                std=[55.539, 55.539, 55.539],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RSARDataset',
        ann_file='data/rsar/train/annfiles/',
        img_prefix='data/rsar/train/images/',
        pipeline=[
            dict(type='LoadImageFromFileMultiExt'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='RResize',
                img_scale=[(640, 640), (800, 800), (960, 960)],
                multiscale_mode='value'),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version='le90'),
            dict(
                type='PolyRandomRotate',
                rotate_ratio=0.5,
                mode='range',
                angles_range=180,
                auto_bound=False,
                version='le90'),
            dict(
                type='Normalize',
                mean=[53.7795, 53.7795, 53.7795],
                std=[55.539, 55.539, 55.539],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        version='le90'),
    val=dict(
        type='RSARDataset',
        ann_file='data/rsar/val/annfiles/',
        img_prefix='data/rsar/val/images/',
        pipeline=[
            dict(type='LoadImageFromFileMultiExt'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[53.7795, 53.7795, 53.7795],
                        std=[55.539, 55.539, 55.539],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'),
    test=dict(
        type='RSARDataset',
        ann_file='data/rsar/test/annfiles/',
        img_prefix='data/rsar/test/images/',
        pipeline=[
            dict(type='LoadImageFromFileMultiExt'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[53.7795, 53.7795, 53.7795],
                        std=[55.539, 55.539, 55.539],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'))
evaluation = dict(interval=1, metric='mAP50', also_eval_on_test=True)
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=30,
        layer_decay_rate=0.9,
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.2))))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    min_lr_ratio=1e-06,
    by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1, max_keep_ckpts=4)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
angle_version = 'le90'
pretrained = 'ckpts/iTPN/checkpoint-1200.pth'
model = dict(
    type='ReDet',
    backbone=dict(
        _delete_=True,
        type='iTPN_pixel',
        img_size=800,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        mlp_depth=3,
        fpn_dim=256,
        fpn_depth=1,
        depth=24,
        num_heads=8,
        bridge_mlp_ratio=3.0,
        mlp_ratio=4.0,
        num_outs=5,
        out_embed_dim=256,
        drop_path_rate=0.05,
        ape=True,
        rpe=False,
        patch_norm=True,
        use_checkpoint=False,
        num_classes=6,
        init_cfg=dict(
            type='Pretrained', checkpoint='ckpts/iTPN/checkpoint-1200.pth')),
    neck=None,
    rpn_head=dict(
        type='RotatedRPNHead',
        in_channels=256,
        feat_channels=256,
        version='le90',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='RoITransRoIHead',
        version='le90',
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
                featmap_strides=[4, 8, 16, 32])
        ],
        bbox_head=[
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=6,
                bbox_coder=dict(
                    type='DeltaXYWHAHBBoxCoder',
                    angle_range='le90',
                    norm_factor=2,
                    edge_swap=True,
                    target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=6,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range='le90',
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
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
            score_thr=0.01,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)))
_data_root = 'data/rsar/'
_img_norm = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)
_test_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[53.7795, 53.7795, 53.7795],
                std=[55.539, 55.539, 55.539],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
work_dir = './work_dirs/redet_itpn_base_3x_rsar'
auto_resume = False
gpu_ids = range(0, 4)
