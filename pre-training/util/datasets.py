# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
"""预训练数据加载工具。

* build_dataset / build_transform：ImageNet 风格（train/val 子目录）数据集；
* MyDataSet / load_data          ：从任意目录递归扫描图片并按子文件夹名打标签；
* SARImageDataset                ：按 txt 文件列表 + 前缀路径映射加载（500K 语料用）。
"""
import os
import time
import re

import PIL
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=PIL.Image.BICUBIC))  # keep ratio w.r.t. 224 images
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class MyDataSet(Dataset):
    """(图片路径列表, 标签列表) -> 灰度读取的 SAR 数据集。"""

    def __init__(self, image_list, label_list, transform=None):
        self.image_list = image_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = self.image_list[index]
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)   # SAR 幅度图按单通道读取
        img = Image.fromarray(img)
        label = self.label_list[index]
        if self.transform:
            img = self.transform(img)
        return img, label


class SARImageDataset(Dataset):
    """SAR 图像数据集，支持多文件列表 + 多前缀路径映射（预训练语料用）。"""

    def __init__(self,
                 file_list_paths,
                 prefix_mappings=None,
                 transform=None,
                 check_files_exist=True):
        """
        Args:
            file_list_paths: 包含图像路径的 txt 文件路径（或路径列表）
            prefix_mappings: 路径前缀映射 {旧前缀: 新前缀} 或 [(旧, 新), ...]
            transform: 图像变换
            check_files_exist: 是否检查文件是否存在
        """
        self.transform = transform
        self.check_files_exist = check_files_exist
        self.prefix_mappings = self._process_prefix_mappings(prefix_mappings)
        self.image_paths = self._load_and_process_file_lists(file_list_paths)

        print(f"数据集加载完成，共 {len(self.image_paths)} 张图像")
        if self.prefix_mappings:
            print(f"应用了 {len(self.prefix_mappings)} 个前缀映射规则")

    def _process_prefix_mappings(self, prefix_mappings):
        if prefix_mappings is None:
            return []
        if isinstance(prefix_mappings, dict):
            return list(prefix_mappings.items())
        elif isinstance(prefix_mappings, list):
            return prefix_mappings
        raise ValueError("prefix_mappings 必须是字典或列表格式")

    def _load_and_process_file_lists(self, file_list_paths):
        if isinstance(file_list_paths, str):
            file_list_paths = [file_list_paths]

        all_paths = []
        for file_list_path in file_list_paths:
            if not os.path.exists(file_list_path):
                print(f"警告: 文件列表不存在 {file_list_path}")
                continue
            with open(file_list_path, 'r', encoding='utf-8') as f:
                paths = [line.strip() for line in f if line.strip()]
                all_paths.extend(paths)

        print(f"从 {len(file_list_paths)} 个文件列表加载了 {len(all_paths)} 个路径")

        processed_paths = self._apply_prefix_mappings(all_paths)

        if self.check_files_exist:
            valid_paths = []
            missing_count = 0
            for path in processed_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    missing_count += 1
                    if missing_count <= 10:
                        print(f"警告: 文件不存在 {path}")
            if missing_count > 10:
                print(f"... 还有 {missing_count - 10} 个文件不存在")
            if len(valid_paths) < len(processed_paths):
                print(f"文件检查: {len(valid_paths)}/{len(processed_paths)} 个文件有效")
            return valid_paths
        return processed_paths

    def _apply_prefix_mappings(self, paths):
        processed_paths = []
        for path in paths:
            new_path = path
            for old_prefix, new_prefix in self.prefix_mappings:
                if new_path.startswith(old_prefix):
                    new_path = new_path.replace(old_prefix, new_prefix, 1)
                    new_path = new_path.replace('\\', '/')
                    break
            processed_paths.append(new_path)
        return processed_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            image = Image.fromarray(image)
            if self.transform:
                image = self.transform(image)
            return image, image_path
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            if self.transform:
                dummy_image = self.transform(Image.new('L', (224, 224), color=0))
            else:
                dummy_image = torch.zeros(1, 224, 224)
            return dummy_image, image_path

    def get_all_paths(self):
        return self.image_paths


def load_data(file_dir, transform):
    """从目录递归扫描所有图片，按子文件夹名作为标签。

    Args:
        file_dir: 预训练数据根目录（每个子文件夹为一个无标签类别）
        transform: 图像变换

    Returns:
        MyDataSet
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp'}

    pic_list = []
    label_list = []

    for root, dirs, files in os.walk(file_dir):
        files = sorted(files)
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext not in image_extensions:
                continue
            pic_list.append(os.path.join(root, file))
            label_list.append(os.path.basename(root))   # 子文件夹名作为标签

    dataset = MyDataSet(pic_list, label_list, transform)
    return dataset
