# =============================================================================
# DDHR-SK semantic segmentation config (reference reproduction)
# UperNet + iTPN-Base (MMSegmentation 1.x / mmengine style)
#
# NOTE / 说明:
# - Archived from the original experiment that produced the paper's DDHR-SK
#   numbers (see the sibling training log in this folder).
# - Uses the custom `DDHRSKDataset`, `iTPN_weijie` backbone and
#   `itpn_LayerDecayOptimizerConstructor`; register them in MMSeg as needed.
# - Absolute paths below must be adapted to your environment.
#
# 此配置为论文 DDHR-SK 结果所用实验配置的存档（训练日志见本目录），
# 依赖自定义 `DDHRSKDataset`、`iTPN_weijie` 骨干与 `itpn_LayerDecayOptimizerConstructor`
# （需在 MMSeg 中注册）。下方绝对路径需按环境修改。
# =============================================================================
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
auto_scale_lr = dict(enable=False, base_batch_size=32)
work_dir = './work_dirs/new_weijie_ddhr_1xb4_80k_1e-3_dp10_ld90'
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[53.7795, 53.7795, 53.7795],
        std=[55.539, 55.539, 55.539],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(256, 256)),
    pretrained=None,
    backbone=dict(
        type='iTPN_weijie',
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-06),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=0.1,
        modality='SAR',
        embed_dim=512,
        mlp_depth1=3,
        mlp_depth2=3,
        depth=24,
        fpn_dim=256,
        fpn_depth=2,
        qkv_bias=True,
        ape=True,
        rpe=False,
        num_outs=4,
        use_checkpoint=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/root/autodl-tmp/weights/weijie_saratrx_2/checkpoint-1200.pth')),
    neck=None,
    decode_head=dict(
        type='UPerHead',
        in_channels=[384, 512, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'DDHRSKDataset'
data_root = '/root/autodl-tmp/DDHRSK/'
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=(512, 128),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='DDHRSKDataset',
        data_root='/root/autodl-tmp/DDHRSK/',
        data_prefix=dict(
            img_path='sar_images_train', seg_map_path='anns_train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='RandomResize',
                scale=(512, 128),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DDHRSKDataset',
        data_root='/root/autodl-tmp/DDHRSK/',
        data_prefix=dict(img_path='sar_images_val', seg_map_path='anns_val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', scale=(256, 256), keep_ratio=True),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DDHRSKDataset',
        data_root='/root/autodl-tmp/DDHRSK/',
        data_prefix=dict(img_path='sar_images_val', seg_map_path='anns_val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', scale=(256, 256), keep_ratio=True),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(
    type='CustomIoUMetric', iou_metrics=['mIoU', 'mFscore', 'Kappa'])
test_evaluator = dict(
    type='CustomIoUMetric', iou_metrics=['mIoU', 'mFscore', 'Kappa'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    clip_grad=dict(max_norm=35, norm_type=2),
    constructor='itpn_LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        decay_type='layer_wise',
        num_layers=31,
        decay_rate=0.9,
        absolute_pos_embed=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0),
        norm=dict(decay_mult=0.0)),
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
        end=2000),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=2000,
        end=80000,
        by_epoch=False)
]
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        max_keep_ckpts=1,
        by_epoch=False,
        interval=1000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
launcher = 'none'
