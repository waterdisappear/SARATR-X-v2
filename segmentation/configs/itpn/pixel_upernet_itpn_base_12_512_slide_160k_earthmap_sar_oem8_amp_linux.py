# EarthMap SAR：与 OpenEarthMap 光学版标签对齐（8 类，trainId 1–8，0 为无效）。
# 目录：<data_root>/{train,val,test}/{sar_images,labels}，磁盘瓦片多为 1024×1024；训练 RandomCrop 512。
# 骨干 img_size=512 与 crop 一致。多卡 DDP 且每卡 batch=1 时 head 须用 SyncBN（见下），否则 BN 在 N=1 训练报错。
# 单卡训练且 batch=1 时：提高 samples_per_gpu 或仅用于 debug。
# Windows 本地：只改 data_root / model.pretrained（或命令行 --cfg-options）。
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# _delete_=True：避免与 default_runtime 的 TextLoggerHook 合并成双日志器
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[dict(type='SafeTextLoggerHook', by_epoch=False)])

crop_size = (512, 512)

# 训练集 OEM 像素频率 sqrt-median 平衡 CE（Bareland≈1.76%→权重≈2.17；主导类≈0.58）。
# eval 指标仍为无权重 mIoU；若训练发散可改为全 1 或整体乘以 0.5。
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

# iTPN-base 预训练（与 air_polarsar2_amp 系列一致）；本地无该路径时可改为 None 或用 --load-from
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
        drop_path_rate=0.1,
        num_outs=4,
        init_values=None),
    decode_head=dict(
        in_channels=[384, 512, 768, 768],
        num_classes=8,
        channels=768,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=_oem8_ce_class_weight)),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=8,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4,
            class_weight=_oem8_ce_class_weight)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

# 与 air_polarsar2_amp 系列一致（三通道输入时的归一化习惯）
img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

dataset_type = 'CustomDataset'
data_root = r'data/earthmap'
train_root = data_root + '/train'
val_root = data_root + '/val'
test_root = data_root + '/test'

# 顺序对应 OEM trainId 1..8 → 训练索引 0..7（配合 LoadAnnotations reduce_zero_label）
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

# 验证/测试：只能 Collect「img」。multi_gpu_test 会把 batch 里所有键传给 model.forward，
# simple_test 不接受 gt_semantic_seg，若在 pipeline 里加载标注会报错。
# 标签对齐：靠下方 data.val/test 的 reduce_zero_label=True，evaluate() 从磁盘读 GT 时再映射。
# ValTest 的 Kappa/OA 见 mmcv_custom/rs_metrics.py（同样用 reduce_zero_label）。
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

# 优化器：默认「全参数同一学习率」作 baseline（避免 iTPN 分层衰减把底层 lr 压得过低，
# 日志里若只打印某一 param group，可能看到远小于 base 的 lr）。
# 若需分层衰减，可改回：constructor='iTPNLayerDecayOptimizerConstructor',
# paramwise_cfg=dict(num_layers=30, layer_decay_rate=0.95)（0.95 比 0.90 对底层更友好）。
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

# ValTest*EvalHook：每 interval 先 val 再 test；GT 由 evaluate() 从 ann_dir 读盘，与训练一致使用
# data.* 的 reduce_zero_label；Kappa/OA 混淆矩阵见 mmcv_custom/rs_metrics.py（同样用 reduce_zero_label）。
# 各类像素占比：python tools/stats_earthmap_label_pixels.py --data-root <EarthMap根目录>
eval_test = True
evaluation = dict(
    _delete_=True,
    interval=2000,
    metric=['mIoU'],
    eval_test=True,
    save_best='mIoU')
