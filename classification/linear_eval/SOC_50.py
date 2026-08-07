"""
SOC-50 分类评测（线性探测）(SOC-50 classification evaluation, linear probing)

对应论文中的 ATRNet-STAR（SOC-50 划分）线性探测评测：
frozen iTPN backbone + 仅训练线性分类头（fc_norm + head）。

This script evaluates linear probing on the SOC-50 split of ATRNet-STAR:
the iTPN backbone is frozen and only the linear head (fc_norm + head) is trained.

用法 (Usage)::

    python SOC_50.py --data_path /path/to/SOC_50classes --classes 50 --epochs 30

数据目录约定 (data layout)::

    <data_path>/
        train/class_0/xxx.png, ...
        test/class_0/xxx.png, ...
"""

import os
import sys
import re
import argparse
import collections
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.DataLoad import load_data
from utils.TrainTest import model_train, model_val, model_test
from model.models_itpn import itpn_base

# 仓库根目录：用于定位 dataset/ 下数据的默认路径
# (repo root, used to derive the default data path under dataset/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parameter_setting():
    parser = argparse.ArgumentParser(description='SOC-50 linear probing evaluation')
    # 默认数据路径指向仓库 dataset/classification/SOC_50classes（可用 --data_path 覆盖）
    default_data = os.path.join(REPO_ROOT, 'dataset', 'classification', 'SOC_50classes') + os.sep
    parser.add_argument('--data_path', type=str, default=default_data,
                        help='data root containing train/ and test/ folders')
    parser.add_argument('--GPU_ids', type=int, default=0, help='GPU id')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--classes', type=int, default=50, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--fold', type=int, default=1, help='number of folds (K-fold)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    return parser.parse_args()


def build_transform():
    """构造评测数据预处理：Resize 到 224 后按 SAR 统计值归一化。

    Build the evaluation transform: resize to 224 and normalize with SAR statistics.
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.2109, 0.2109, 0.2109), (0.2178, 0.2178, 0.2178)),
    ])


def main():
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)

    history = collections.defaultdict(list)  # 记录每一折的指标 (metrics per fold)

    data_transform = build_transform()

    # 加载训练 / 测试集 (load train and test sets)
    train_all = load_data(arg.data_path + 'train', data_transform)
    test_set = load_data(arg.data_path + 'test', data_transform)

    for k_F in tqdm(range(arg.fold)):
        train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)

        # 构造冻结骨干的 iTPN-B 并加载预训练权重（见 model/models_itpn.py）
        model = itpn_base(arg.classes)

        opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arg.epochs)
        best_test_accuracy = 0

        for epoch in range(1, arg.epochs + 1):
            print("##### " + str(k_F + 1) + " EPOCH " + str(epoch) + "#####")
            loss = model_train(model=model, data_loader=train_loader, opt=opt, sch=scheduler)
            if epoch % 10 == 0:
                accuracy = model_val(model, test_loader)
                print("Val Accuracy is:{:.2f} %: ".format(accuracy))

        acc = model_test(model, test_loader)
        print('test accuracy is {}'.format(acc))
        history['accuracy'].append(acc)
        print('The best epoch is {}, val accuracy is {}, test accuracy is {}'.
              format(epoch, best_test_accuracy, acc))

    print('OA is {}, STD is {}'.format(np.mean(history['accuracy']), np.std(history['accuracy'])))
    print(history['accuracy'])


if __name__ == '__main__':
    main()
