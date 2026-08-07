# RSAR dataset settings
# RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark
# Ref: https://github.com/zhasion/RSAR
# 目录结构: $data_root/{train,val,test}/{annfiles,images}
# SAR 图像归一化（参考 pixel_itpn_GFL_sardet100k，适用于 SAR）
dataset_type = 'RSARDataset'
data_root = 'data/rsar/'
img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),  # 支持 .bmp/.png/.jpg 等混合后缀
    dict(type='LoadAnnotations', with_bbox=True),
    # 多尺度以 RSAR 基准 800 为中心（RResize 不支持 keep_ratio，已省略）
    dict(
        type='RResize',
        img_scale=[(640, 640), (800, 800), (960, 960)],
        multiscale_mode='value'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(type='Normalize', **img_norm_cfg),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline,
        version='le90'),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline,
        version='le90'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version='le90'))
