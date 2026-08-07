"""
SAR-VSA 分类评测（线性探测）(SAR-VSA classification evaluation, linear probing)

SAR-VSA（数据来自 SARATR-X v1）在冻结 iTPN 骨干上训练线性分类头进行评测。

This script evaluates SAR-VSA with linear probing: a frozen iTPN backbone
plus a linear head (fc_norm + head) trained on top.

用法 (Usage)::

    python SARVASA.py --data_path /path/to/SARVASA --classes 25 --epochs 30
"""

import os
import sys
import argparse
import collections

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.DataLoad import load_data
from utils.TrainTest import model_train, model_val, model_test
from model.models_itpn import itpn_base

# 仓库根目录：用于定位 dataset/ 下数据的默认路径
# (repo root, used to derive the default data path under dataset/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parameter_setting():
    parser = argparse.ArgumentParser(description='SAR-VSA linear probing evaluation')
    default_data = os.path.join(REPO_ROOT, 'dataset', 'classification', 'SARVASA') + os.sep
    parser.add_argument('--data_path', type=str, default=default_data,
                        help='data root containing train/ and test/ folders')
    parser.add_argument('--GPU_ids', type=int, default=0, help='GPU id')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--classes', type=int, default=25, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--fold', type=int, default=1, help='number of folds (K-fold)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    return parser.parse_args()


def build_transform():
    """构造评测数据预处理：Resize + CenterCrop 到 224，按 SAR 统计值归一化。

    Build the evaluation transform: resize + center-crop to 224, normalized with SAR statistics.
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.2109, 0.2109, 0.2109), (0.2178, 0.2178, 0.2178)),
    ])


def main():
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)

    history = collections.defaultdict(list)  # 记录每一折的指标 (metrics per fold)
    data_transform = build_transform()

    train_all = load_data(arg.data_path + 'train', data_transform)
    test_set = load_data(arg.data_path + 'test', data_transform)

    for k_F in tqdm(range(arg.fold)):
        train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)

        # 冻结骨干的 iTPN-B + 预训练权重（见 model/models_itpn.py）
        model = itpn_base(arg.classes)

        opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arg.epochs)
        best_test_accuracy = 0

        for epoch in tqdm(range(1, arg.epochs + 1)):
            print("##### " + str(k_F + 1) + " EPOCH " + str(epoch) + "#####")
            loss = model_train(model=model, data_loader=train_loader, opt=opt, sch=scheduler)

        acc = model_test(model, test_loader)
        print('test accuracy is {}'.format(acc))
        history['accuracy'].append(acc)
        print('The best epoch is {}, val accuracy is {}, test accuracy is {}'.
              format(epoch, best_test_accuracy, acc))

    print('OA is {}, STD is {}'.format(np.mean(history['accuracy']), np.std(history['accuracy'])))
    print(history['accuracy'])


if __name__ == '__main__':
    main()
