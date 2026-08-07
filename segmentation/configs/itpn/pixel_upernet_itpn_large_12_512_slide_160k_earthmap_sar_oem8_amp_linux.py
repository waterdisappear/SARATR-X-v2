# EarthMap SAR（large iTPN）：与 OpenEarthMap 光学版标签对齐（8 类，trainId 1–8，0 为无效）。
# 数据/损失/优化等非 large 专属项与 pixel_upernet_itpn_base_12_512_slide_160k_earthmap_sar_oem8_amp_linux.py 对齐；
# large 专属：pretrained、embed_dim/depth/heads、decode/aux in_channels、rpe/use_checkpoint/drop_path 等保持本文件取值。
# 目录：<data_root>/{train,val,test}/{sar_images,labels}，磁盘瓦片多为 1024×1024，训练 RandomCrop 512。
# Windows 本地：只改 data_root / model.pretrained（或命令行 --cfg-options）。
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[dict(type='SafeTextLoggerHook', by_epoch=False)])

crop_size = (512, 512)

# 训练集 OEM 像素频率 sqrt-median 平衡 CE（与 base EarthMap 一致）。
_oem8_ce_class_weight = [
    2.165,
    0.582,
    0.598,
    1.057,
    0.763,
    0.873,
    1.105,
    0.858,
]

model = dict(
    pretrained='ckpts/itpn_large/checkpoint-1200.pth',
    backbone=dict(
        type='iTPN',
        img_size=512,
        patch_size=16,
        embed_dim=768,
        mlp_depth1=2,
        mlp_depth2=2,
        depth=40,
        num_heads=12,
        mlp_ratio=4,
        fpn_dim=256,
        fpn_depth=2,
        qkv_bias=True,
        ape=True,
        rpe=False,
        use_checkpoint=True,
        drop_path_rate=0.2,
        num_outs=4,
        init_values=None),
    decode_head=dict(
        in_channels=[448, 640, 1024, 1024],
        num_classes=8,
        channels=768,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=_oem8_ce_class_weight)),
    auxiliary_head=dict(
        in_channels=1024,
        num_classes=8,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=_oem8_ce_class_weight)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

dataset_type = 'CustomDataset'
data_root = r'data/earthmap'
# data_root = r'D:\Data\SAR\地物\EarthMap'
train_root = data_root + '/train'
val_root = data_root + '/val'
test_root = data_root + '/test'

classes = (
    'Bareland',
    'Rangeland',
    'Developed Space',
    'Road',
    'Tree',
    'Water',
    'Agriculture Land',
    'Building',
)
palette = [
    [128, 0, 0],
    [0, 255, 36],
    [148, 148, 148],
    [255, 255, 255],
    [34, 97, 38],
    [0, 69, 255],
    [75, 181, 73],
    [222, 31, 7],
]

train_pipeline = [
    dict(type='LoadPreprocessedGrayAs3Ch'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.8, 1.2)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.90),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadPreprocessedGrayAs3Ch'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=train_root,
        img_dir='sar_images',
        ann_dir='labels',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        reduce_zero_label=True,
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=val_root,
        img_dir='sar_images',
        ann_dir='labels',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        reduce_zero_label=True,
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=test_root,
        img_dir='sar_images',
        ann_dir='labels',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        reduce_zero_label=True,
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=5e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=3000,
    warmup_ratio=1e-5,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)

runner = dict(_delete_=True, type='IterBasedRunnerAmp', max_iters=160000)
fp16 = None
optimizer_config = dict(
    _delete_=True,
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)

# ValTest*EvalHook；各类像素占比：python tools/stats_earthmap_label_pixels.py --data-root <EarthMap根目录>
eval_test = True
evaluation = dict(
    _delete_=True,
    interval=2000,
    metric=['mIoU'],
    eval_test=True,
    save_best='mIoU')
