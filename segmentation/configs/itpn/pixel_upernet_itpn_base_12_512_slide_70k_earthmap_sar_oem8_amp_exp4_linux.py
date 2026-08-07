# EarthMap SAR — 实验4：CE+Focal（Bareland class_weight=10）+ 50% Bareland 过采样；无 save_best。
# 依赖：mmcv_custom.CEFocalLoss、EarthMapOEM8OversampleDataset（启动前 import mmcv_custom）。
# 迭代 70k（日志显示 60k 后收益有限）；多卡请用 SyncBN；FP16 需安装 apex。
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[dict(type='SafeTextLoggerHook', by_epoch=False)])

crop_size = (512, 512)

# Bareland 强化 + 其余类沿用 sqrt-median 量级（与讨论表一致）
_oem8_exp4_class_weight = [
    10.0,
    0.58,
    0.60,
    1.06,
    0.76,
    0.87,
    1.11,
    0.86,
]

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
        rpe=True,
        drop_path_rate=0.2,
        num_outs=4,
        init_values=None,
        use_checkpoint=False),
    decode_head=dict(
        in_channels=[384, 512, 768, 768],
        num_classes=8,
        channels=768,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(
            type='CEFocalLoss',
            use_sigmoid=False,
            gamma=2.0,
            alpha=0.25,
            class_weight=_oem8_exp4_class_weight,
            ce_weight=1.0,
            focal_weight=1.0,
            loss_weight=1.0)),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=8,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(
            type='CEFocalLoss',
            use_sigmoid=False,
            gamma=2.0,
            alpha=0.25,
            class_weight=_oem8_exp4_class_weight,
            ce_weight=1.0,
            focal_weight=1.0,
            loss_weight=0.4)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

dataset_type_train = 'EarthMapOEM8OversampleDataset'
dataset_type_eval = 'CustomDataset'
data_root = r'data/earthmap'
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
    dict(type='RandomRotate', prob=0.5, degree=10, pad_val=0, seg_pad_val=255),
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
        type=dataset_type_train,
        bareland_oversample_prob=0.5,
        min_bareland_pixels=100,
        bareland_train_id=1,
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
        type=dataset_type_eval,
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
        type=dataset_type_eval,
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
    weight_decay=0.05,
    constructor='iTPNLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=30, layer_decay_rate=0.95))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1e-5,
    power=0.9,
    min_lr=0.0,
    by_epoch=False)

runner = dict(_delete_=True, type='IterBasedRunnerAmp', max_iters=70000)
checkpoint_config = dict(_delete_=True, by_epoch=False, interval=2000)
fp16 = None
optimizer_config = dict(
    _delete_=True,
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)

eval_test = True
evaluation = dict(
    _delete_=True,
    interval=2000,
    metric=['mIoU'],
    eval_test=True)
