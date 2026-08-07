# FAIR-CSAR split dataset config (trainval + test)

import glob
import os

from mmrotate.datasets.dota import collect_dota_class_names_from_ann_folder


def _env_or_default(name, default):
    """读取环境变量，为空时回退到默认路径 (read env var, fallback to default)."""
    val = os.environ.get(name, '').strip()
    return val if val else default


# 数据根目录：优先环境变量，否则使用默认相对路径 data/faircsar/...
# (Data root: env var first, then relative path under mmrotate data/.)
_train_env = _env_or_default('FAIRCSAR_SL_TRAIN', 'data/faircsar/SL-TRAINVAL/trainval/')
_test_env = _env_or_default('FAIRCSAR_SL_TEST', 'data/faircsar/SL-TRAINVAL/test/')
_root_train = _train_env.rstrip('/\\')
_root_test = _test_env.rstrip('/\\')

_ann_train_dir = os.path.join(_root_train, 'Annotations')
_img_train_dir = os.path.join(_root_train, 'PNGImages')

_ann_test_dir = os.path.join(_root_test, 'Annotations')
_img_test_dir = os.path.join(_root_test, 'PNGImages')

angle_version = 'le90'

img_norm_cfg = dict(
    mean=[53.7795, 53.7795, 53.7795],
    std=[55.539, 55.539, 55.539],
    to_rgb=True)


# 数据目录存在性检查：不存在时打印警告而非抛异常，便于先浏览配置
# (Warn instead of assert so the config can be inspected without data.)
if not os.path.isdir(_ann_train_dir):
    print(f'[WARN][FAIRCSAR] train_ann_dir 不存在: {_ann_train_dir}')
if not os.path.isdir(_ann_test_dir):
    print(f'[WARN][FAIRCSAR] test_ann_dir 不存在: {_ann_test_dir}')

# 与 DOTADataset.load_annotations 一致
if os.path.isdir(_ann_train_dir) and len(glob.glob(os.path.join(_ann_train_dir, '*.txt'))) > 0:
    _cls_train = set(collect_dota_class_names_from_ann_folder(_ann_train_dir, angle_version))
else:
    _cls_train = set()
if os.path.isdir(_ann_test_dir) and len(glob.glob(os.path.join(_ann_test_dir, '*.txt'))) > 0:
    _cls_test = set(collect_dota_class_names_from_ann_folder(_ann_test_dir, angle_version))
else:
    _cls_test = set()
CLASSES = tuple(sorted(_cls_train | _cls_test))
num_classes = len(CLASSES) if CLASSES else 6
_label_union_dirs = [_ann_train_dir, _ann_test_dir]


train_pipeline = [
    dict(type='LoadImageFromFileMultiExt'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RResize',
        img_scale=[(640, 640), (800, 800), (960, 960)],
        multiscale_mode='value'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
            dict(type='Collect', keys=['img']),
        ]),
]

dataset_type = 'FAIRCSARDataset'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=_ann_train_dir,
        img_prefix=_img_train_dir,
        pipeline=train_pipeline,
        version=angle_version,
        label_union_dirs=_label_union_dirs,
        split='train',
        val_ratio=0.1,
        seed=42),
    # If you have a dedicated val split, replace this with your val folders.
    val=dict(
        type=dataset_type,
        ann_file=_ann_train_dir,
        img_prefix=_img_train_dir,
        pipeline=test_pipeline,
        version=angle_version,
        label_union_dirs=_label_union_dirs,
        split='val',
        val_ratio=0.1,
        seed=42),
    test=dict(
        type=dataset_type,
        ann_file=_ann_test_dir,
        img_prefix=_img_test_dir,
        pipeline=test_pipeline,
        version=angle_version,
        label_union_dirs=_label_union_dirs),
)

