# ReDet + iTPN 骨干（官方 det 版）+ RSAR 数据集（Large）
# 骨干参考: https://github.com/sunsmarterjie/iTPN/tree/main/det
# 数据集: https://github.com/zhasion/RSAR
# Large Backbone 参数参考: pixel_itpn_GFL_large.py
_base_ = [
    '../_base_/datasets/rsar.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le90'
# 官方 iTPN det 预训练权重（pixel itpn large），需放置到 ckpts/iTPN_large/
pretrained = 'ckpts/iTPN_large/checkpoint-1200.pth'
# iTPN_pixel 的 img_size 仅决定 __init__ 时 absolute_pos_embed 网格（官方 / checkpoint-1200 为 224→14×14）
# 检测实际输入分辨率由数据 pipeline 决定（如 800/1024）；前向会对 pos 做插值适配
itpn_img_size = 800

model = dict(
    type='ReDet',
    backbone=dict(
        _delete_=True,
        type='iTPN_pixel',
        img_size=itpn_img_size,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        mlp_depth1=2,
        mlp_depth2=2,
        fpn_dim=256,
        fpn_depth=1,
        depth=40,
        num_heads=12,
        bridge_mlp_ratio=3.,
        mlp_ratio=4.,
        num_outs=5,
        out_embed_dim=256,
        drop_path_rate=0.2,
        ape=True,
        rpe=False,
        patch_norm=True,
        use_checkpoint=True,
        num_classes=6,
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
            ratios=[0.25, 0.5, 1.0, 2.0, 4.0],  # 针对 RSAR 细长目标
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
                num_classes=6,
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
                num_classes=6,
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
            # 略降阈值提召回；max_per_img 抑制低分长尾
            score_thr=0.03,
            # 注：mmrotate ``multiclass_nms_rotated`` 仅走 ``nms_rotated``，读 ``nms.iou_thr``；
            # 无旋转 soft_nms。此处用更紧的 IoU（0.55）近似「更强去重」诉求。
            nms=dict(iou_thr=0.55),
            max_per_img=300)))

# 优化器：Large 结构对齐 LayerDecay（mlp_depth1+mlp_depth2+depth=44）
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=44,
        layer_decay_rate=0.95,
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
        }))

optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2))

# 3x 36 epoch，CosineAnnealing（保持 max_epochs=36；略加长 warmup、略抬高余弦末端）
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=0.001,
    min_lr_ratio=5e-6,
    by_epoch=True)

runner = dict(type='EpochBasedRunner', max_epochs=36)

# 评估：只输出 mAP50，val + test 都看
# COCO 风格 mAP：在 IoU 0.5:0.05:0.95 共 10 个阈值上算各类 AP 再平均，返回的 mAP 为这 10 个均值的平均
_coco_iou_thr = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
evaluation = dict(
    interval=1,
    metric='mAP',
    iou_thr=_coco_iou_thr,
    also_eval_on_test=True)
# 每轮保存权重，最多保留 4 个
checkpoint_config = dict(interval=1, max_keep_ckpts=4)

# 显式写全 data（含 PolyRandomRotate），避免覆盖 base 后缺 val/test 导致 cfg.data 报错
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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RSARDataset',
        ann_file=_data_root + 'train/annfiles/',
        img_prefix=_data_root + 'train/images/',
        pipeline=[
            dict(type='LoadImageFromFileMultiExt'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(800, 800)),
            dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25], direction=['horizontal', 'vertical', 'diagonal'], version='le90'),
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

# 可选：混合精度
# fp16 = dict(loss_scale=512.0)
auto_scale_lr = dict(enable=True, base_batch_size=8)
