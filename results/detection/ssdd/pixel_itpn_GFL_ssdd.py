# =============================================================================
# SSDD horizontal-bbox detection config (reference reproduction)
# GFL + iTPN-Base backbone (MMDetection style)
#
# NOTE / 说明:
# - Archived from the original experiment that produced the paper's SSDD numbers.
# - Targets MMDetection with the `iTPN_pixel` backbone (see `detection/`).
# - Absolute paths below must be adapted to your environment.
#
# 此配置为论文 SSDD 结果所用实验配置的存档（训练日志见本目录），
# 风格为 MMDetection + `iTPN_pixel` 骨干。下方绝对路径需按环境修改。
# =============================================================================
dataset_type = 'CocoDataset'
data_root = '/mnt/data7t/liweijie19/code/detection_iptn/data/SSDD/'
img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale': [(384, 384), (448, 448), (512, 512), (576, 576),
                          (640, 640), (704, 704), (768, 768), (800, 800)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type':
                      'Resize',
                      'img_scale': [(448, 448), (512, 512), (576, 576),
                                    (640, 640)],
                      'multiscale_mode':
                      'value',
                      'keep_ratio':
                      True
                  }, {
                      'type':
                      'Resize',
                      'img_scale': [(576, 576), (640, 640), (704, 704),
                                    (768, 768), (800, 800)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
    dict(
        type='Normalize',
        mean=[53.7795, 53.7795, 53.7795],
        std=[55.539, 55.539, 55.539],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[53.7795, 53.7795, 53.7795],
                std=[55.539, 55.539, 55.539],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file=
        '/mnt/data7t/liweijie19/code/detection_iptn/data/SSDD/annotations/train.json',
        img_prefix=
        '/mnt/data7t/liweijie19/code/detection_iptn/data/SSDD/images/train/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type':
                    'Resize',
                    'img_scale':
                    [(384, 384), (448, 448), (512, 512), (576, 576),
                     (640, 640), (704, 704), (768, 768), (800, 800)],
                    'multiscale_mode':
                    'value',
                    'keep_ratio':
                    True
                }],
                          [{
                              'type':
                              'Resize',
                              'img_scale': [(448, 448), (512, 512), (576, 576),
                                            (640, 640)],
                              'multiscale_mode':
                              'value',
                              'keep_ratio':
                              True
                          }, {
                              'type':
                              'Resize',
                              'img_scale': [(576, 576), (640, 640), (704, 704),
                                            (768, 768), (800, 800)],
                              'multiscale_mode':
                              'value',
                              'override':
                              True,
                              'keep_ratio':
                              True
                          }]]),
            dict(
                type='Normalize',
                mean=[53.7795, 53.7795, 53.7795],
                std=[55.539, 55.539, 55.539],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        filter_empty_gt=True),
    val=dict(
        type='CocoDataset',
        ann_file=
        '/mnt/data7t/liweijie19/code/detection_iptn/data/SSDD/annotations/test.json',
        img_prefix=
        '/mnt/data7t/liweijie19/code/detection_iptn/data/SSDD/images/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[53.7795, 53.7795, 53.7795],
                        std=[55.539, 55.539, 55.539],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        ann_file=
        '/mnt/data7t/liweijie19/code/detection_iptn/data/SSDD/annotations/test.json',
        img_prefix=
        '/mnt/data7t/liweijie19/code/detection_iptn/data/SSDD/images/test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[53.7795, 53.7795, 53.7795],
                        std=[55.539, 55.539, 55.539],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    persistent_workers=True,
    pin_memory=True)
evaluation = dict(
    interval=1,
    metric='bbox',
    classwise=True,
    proposal_nums=(100, 300, 1000),
    gpu_collect=True)
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=30,
        layer_decay_rate=0.9,
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr_ratio=1e-07,
    by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(
    interval=1, max_keep_ckpts=5, save_optimizer=True, by_epoch=True)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=True, base_batch_size=8)
pretrained = '/mnt/data7t/liweijie19/code/detection_iptn/ckpt/jiaquan_simple/checkpoint-1200.pth'
model = dict(
    type='GFL',
    backbone=dict(
        _delete_=True,
        type='iTPN_pixel',
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=512,
        mlp_depth1=3,
        mlp_depth2=3,
        fpn_dim=256,
        fpn_depth=1,
        depth=24,
        num_heads=8,
        bridge_mlp_ratio=3.0,
        mlp_ratio=4.0,
        num_outs=5,
        out_embed_dim=256,
        drop_path_rate=0.05,
        num_classes=1,
        ape=True,
        rpe=False,
        patch_norm=True,
        use_checkpoint=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/mnt/data7t/liweijie19/code/detection_iptn/ckpt/jiaquan_simple/checkpoint-1200.pth'
        )),
    neck=None,
    bbox_head=dict(
        type='GFLHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
            reduction='mean'),
        loss_dfl=dict(
            type='DistributionFocalLoss', loss_weight=0.25, reduction='mean'),
        loss_bbox=dict(type='GIoULoss', loss_weight=6.0, reduction='mean'),
        reg_max=16,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9, ignore_iof_thr=0.5),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=4000,
        min_bbox_size=2,
        score_thr=0.03,
        nms=dict(type='soft_nms', iou_threshold=0.55, method='gaussian'),
        max_per_img=300))
fp16 = dict(loss_scale=512.0)
seed = 42
deterministic = True
cudnn_benchmark = True
work_dir = './work_dirs/pixel_itpn_GFL_ssdd'
auto_resume = False
gpu_ids = range(0, 6)
