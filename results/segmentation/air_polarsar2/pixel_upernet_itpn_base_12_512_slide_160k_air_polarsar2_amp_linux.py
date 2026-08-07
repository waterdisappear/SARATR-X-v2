# =============================================================================
# AIR-PolSAR-Seg-2.0 semantic segmentation config (reference reproduction)
# UperNet + iTPN-Base (MMSegmentation style)
#
# NOTE / 说明:
# - Archived from the original experiment that produced the paper's
#   AIR-PolSAR-Seg-2.0 numbers (see the sibling training log in this folder).
# - A maintained copy of this config lives in `segmentation/configs/itpn/`.
# - Absolute paths below must be adapted to your environment.
#
# 此配置为论文 AIR-PolSAR-Seg-2.0 结果所用实验配置的存档（训练日志见本目录）；
# `segmentation/configs/itpn/` 下有可维护的同名配置。下方绝对路径需按环境修改。
# =============================================================================
# base config reference (dump format) / base 配置引用（序列化格式）：
# /mnt/data7t/lwj/mmseg_lwj/configs/_base_/models/upernet_beit.py
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='XCiT',
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# base config reference: /mnt/data7t/lwj/mmseg_lwj/configs/_base_/default_runtime.py
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

# base config reference: /mnt/data7t/lwj/mmseg_lwj/configs/_base_/schedules/schedule_160k.py
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU')

# base config reference: /mnt/data7t/lwj/mmseg_lwj/configs/itpn/pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux.py
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

log_config = dict(
    interval=50, hooks=[dict(type='SafeTextLoggerHook', by_epoch=False)])

crop_size = (512, 512)

model = dict(
    pretrained='/mnt/data7t/lwj/mmseg_lwj/ckpt/base/checkpoint-1200.pth',
    backbone=dict(
        type='iTPN',
        img_size=512,
        patch_size=16,
        embed_dim=512,
        mlp_depth1=3,
        mlp_depth2=3,
        depth=24,
        num_heads=8,
        mlp_ratio=4,
        fpn_dim=256,
        fpn_depth=2,
        qkv_bias=True,
        ape=True,
        rpe=True,
        drop_path_rate=0.1,
        num_outs=4),
    decode_head=dict(
        in_channels=[384, 512, 768, 768],
        num_classes=6,
        channels=768,
        norm_cfg=dict(type='BN', requires_grad=True)),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

dataset_type = 'CustomDataset'
data_root = '/mnt/data7t/lwj/mmseg_lwj/data/AIR-PolarSAR-Seg-2.0'
train_root = f'{data_root}/train'
val_root = f'{data_root}/val'

classes = (
    'housing', 'industrial', 'natural', 'land_use', 'water', 'other')
palette = [
    [255, 255, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [0, 255, 255],
    [255, 255, 255],
]

train_pipeline = [
    dict(type='LoadPolSARAmplitudeRGB'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadPolSARAmplitudeRGB'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=train_root,
        img_dir='hh',
        ann_dir='gt',
        img_suffix='_hh_amp.tiff',
        seg_map_suffix='.tiff',
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=val_root,
        img_dir='hh',
        ann_dir='gt',
        img_suffix='_hh_amp.tiff',
        seg_map_suffix='.tiff',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=val_root,
        img_dir='hh',
        ann_dir='gt',
        img_suffix='_hh_amp.tiff',
        seg_map_suffix='.tiff',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='iTPNLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=0.90))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='IterBasedRunnerAmp')
fp16 = None
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)

evaluation = dict(interval=2000, metric='mIoU')
