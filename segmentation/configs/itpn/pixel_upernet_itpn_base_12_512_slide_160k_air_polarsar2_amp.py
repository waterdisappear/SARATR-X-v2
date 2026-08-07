_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

log_config = dict(
    interval=50, hooks=[dict(type='SafeTextLoggerHook', by_epoch=False)])

crop_size = (512, 512)

model = dict(
    pretrained='ckpts/itpn_base/checkpoint-1200.pth',
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
        rpe=False,
        drop_path_rate=0.1,
        num_outs=4,
        init_values=None),
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
train_root = r'data/air-polarsar-seg-2.0/train'
val_root = r'data/air-polarsar-seg-2.0/val'

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
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05)

lr_config = dict(
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
