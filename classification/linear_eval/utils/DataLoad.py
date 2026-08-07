"""
数据加载工具 (Data loading utilities)

线性评测（linear evaluation）使用的数据加载接口：
  - load_data                      : 使用 torchvision ImageFolder 从目录加载数据集
                                     (Load a dataset from a directory with torchvision ImageFolder)
  - load_data_with_class_mapping   : 使用训练集的类别映射加载测试集，保证各测试子目录的
                                     标签索引与训练集一致
                                     (Load a test set with the training set's class mapping so that
                                     label indices stay consistent across test subdirectories)

数据集目录约定 (Dataset directory convention)::

    <data_root>/
        train/
            class_0/xxx.png
            class_1/xxx.png
        test/
            class_0/xxx.png
            ...
"""

import torchvision.datasets as datasets


def load_data(file_dir, transform):
    """从 ``file_dir`` 加载 ImageFolder 数据集。

    Load an ImageFolder dataset from ``file_dir``.

    Args:
        file_dir (str): 数据集目录（含 train/test 等子目录）
        transform: torchvision 数据预处理流水线
    Returns:
        torchvision.datasets.ImageFolder
    """
    return datasets.ImageFolder(file_dir, transform=transform)


def load_data_with_class_mapping(file_dir, transform, class_to_idx):
    """加载测试集，并把类别索引强制映射到给定的 ``class_to_idx``（通常来自训练集）。

    Load a test set but remap its class indices to the given ``class_to_idx``
    (typically taken from the training set), so that labels remain consistent
    across different test folders.

    Args:
        file_dir (str): 测试集目录
        transform: torchvision 数据预处理流水线
        class_to_idx (dict): 训练集的 {类别名: 索引} 映射
    Returns:
        torchvision.datasets.ImageFolder: 已重映射标签的数据集
    """
    ds = datasets.ImageFolder(file_dir, transform=transform)

    remapped_samples = []
    missing = set()
    for path, local_idx in ds.samples:
        class_name = ds.classes[local_idx]
        if class_name not in class_to_idx:
            missing.add(class_name)
            continue
        remapped_samples.append((path, int(class_to_idx[class_name])))

    if missing:
        raise ValueError(
            f"Classes in '{file_dir}' not found in provided class_to_idx: {sorted(missing)}"
        )

    ds.samples = remapped_samples
    ds.targets = [t for _, t in remapped_samples]
    ds.class_to_idx = dict(class_to_idx)
    return ds
