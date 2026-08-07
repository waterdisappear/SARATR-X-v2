# WHU-OPT-SAR（large iTPN）：在 pixel_upernet_itpn_base_12_512_slide_160k_whu_opt_sar_sar_linux 上
# 换 large 骨干与 UPer 头通道；数据/归一化/评估逻辑与 base 版一致。
# 预训练权重路径与 pixel_upernet_itpn_large_12_512_slide_160k_air_polarsar2_amp_linux 对齐（large）。
#
# 超参相对「与 base 完全相同」的调整要点（小数据 + 大模型微调）：
# - 略降 lr、拉长 warmup，减轻早期 OA 偏低；
# - 略降 drop_path / wd，减轻过正则导致的整体准确率吃亏；
# - 梯度裁剪抑制 ViT 大模型偶发不稳。
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# 路径：按机器只改下面一行。
data_root = 'data/whu_opt_sar_mmseg_256_split'
# data_root = r'D:\Data\Multimodal\whu-opt-sar\whu_opt_sar_mmseg_256_split'

log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[dict(type='SafeTextLoggerHook', by_epoch=False)])

crop_size = (256, 256)

model = dict(
    pretrained='ckpts/itpn_large/checkpoint-1200.pth',
    backbone=dict(
        type='iTPN',
        img_size=256,
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
        rpe=True,
        use_checkpoint=True,
        drop_path_rate=0.15,
        num_outs=4),
    decode_head=dict(
        in_channels=[448, 640, 1024, 1024],
        num_classes=7,
        channels=768,
        norm_cfg=dict(type='BN', requires_grad=True)),
    auxiliary_head=dict(
        in_channels=1024,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=True)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(256, 256)))

img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)

dataset_type = 'CustomDataset'

classes = (
    'farmland',
    'city',
    'village',
    'water',
    'forest',
    'road',
    'others',
)
palette = [
    [128, 160, 72],
    [255, 64, 64],
    [255, 180, 64],
    [64, 128, 255],
    [48, 128, 48],
    [255, 255, 80],
    [180, 180, 180],
]

train_pipeline = [
    dict(type='LoadPreprocessedGrayAs3Ch'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.95),
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
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# large 显存更大，单卡 batch 不够时可改为 1
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/test',
        ann_dir='ann_dir/test',
        img_suffix='.png',
        seg_map_suffix='.png',
        classes=classes,
        palette=palette,
        pipeline=test_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-5,
    betas=(0.9, 0.999),
    weight_decay=0.03,
    constructor='iTPNLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=44, layer_decay_rate=0.93))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

runner = dict(type='IterBasedRunnerAmp')
fp16 = None
optimizer_config = dict(
    type='DistOptimizerHook',
    update_interval=1,
    grad_clip=dict(max_norm=1.0, norm_type=2),
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True)

eval_test = True
evaluation = dict(
    _delete_=True,
    interval=2000,
    metric=['mIoU'],
    eval_test=True,
    save_best='mIoU')
