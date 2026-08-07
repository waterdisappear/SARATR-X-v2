# WHU-OPT-SAR：仅用 SAR（256 切片经脚本生成后的 MMSeg 目录）。
# 目录结构：<split_root>/img_dir/{train,val,test} 与 <split_root>/ann_dir/{train,val,test}。
# 输入尺寸 256；归一化与 Air-PolSAR 预训练一致（见 pixel_upernet_itpn_base_12_512_slide_160k_air_polarsar2_amp_linux）。
# 训练过程评估：验证集 + 测试集（需 evaluation 中 eval_test=True），指标含 mIoU、aAcc、OA、Kappa。
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# 路径：按机器只改下面一行（与仓库内其它 *linux* 配置同一服务器根目录习惯）。
# 服务器：
data_root = 'data/whu_opt_sar_mmseg_256_split'
# Windows 本地示例：
# data_root = r'D:\Data\Multimodal\whu-opt-sar\whu_opt_sar_mmseg_256_split'

# _delete_=True：避免与 default_runtime 的 TextLoggerHook 合并成「双日志器」
log_config = dict(
    _delete_=True,
    interval=100,
    hooks=[dict(type='SafeTextLoggerHook', by_epoch=False)])

crop_size = (256, 256)

model = dict(
    pretrained='ckpts/itpn_base/checkpoint-1200.pth',
    backbone=dict(
        type='iTPN',
        img_size=256,
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
        num_classes=7,
        channels=768,
        norm_cfg=dict(type='BN', requires_grad=True)),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=7,
        norm_cfg=dict(type='BN', requires_grad=True)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(256, 256)))

# 与 air_polarsar2_amp_linux 预训练一致（三通道伪 SAR）。
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

# eval_test：同时跑验证集与测试集（见 mmcv_custom/val_test_eval_hooks.py）。
# 验证集指标键名为 mIoU / mAcc / aAcc / OA / Kappa；测试集为 test_* 前缀。
# 顶层 eval_test：train_api 在 evaluation 合并异常时仍可作为备用开关。
eval_test = True
evaluation = dict(
    _delete_=True,
    interval=2000,
    metric=['mIoU'],
    eval_test=True,
    save_best='mIoU')
