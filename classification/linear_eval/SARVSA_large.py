"""
SAR-VSA 评测脚本 —— iTPN-Large 变体 (SAR-VSA evaluation with iTPN-Large).

在 SARVASA.py 基础上将骨干由 itpn_base（512 / depth24）改为 itpn_large
（embed_dim=768, depth=40, num_heads=12, 见 model/models_itpn.py）。

Compared to SARVASA.py, only the backbone is switched from itpn_base
to itpn_large (see model/models_itpn.py).
"""
import os
import argparse
import collections

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.DataLoad import load_data
from utils.TrainTest import model_train, model_test
from model.models_itpn import itpn_large

# 仓库根目录：用于定位 dataset/ 下数据的默认路径
# (repo root, used to derive the default data path under dataset/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parameter_setting():
    parser = argparse.ArgumentParser(description='iTPN-Large on SAR-VSA')
    default_data = os.path.join(REPO_ROOT, 'dataset', 'classification', 'SARVASA') + os.sep
    parser.add_argument('--data_path', type=str, default=default_data,
                        help='data root containing train/ and test/ folders')
    parser.add_argument('--GPU_ids', type=int, default=0, help='GPU ids')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--classes', type=int, default=25, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size (Large 模型显存占用更大，默认略小于 base 的 128)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--fold', type=int, default=1, help='K-fold')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    return parser.parse_args()


def main():
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)
    history = collections.defaultdict(list)

    # 评测数据预处理（SAR 统计值归一化）
    # Evaluation transform (normalization with SAR statistics)
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.2109, 0.2109, 0.2109), (0.2178, 0.2178, 0.2178))
    ])

    train_all = load_data(arg.data_path + 'train', data_transform)
    test_set = load_data(arg.data_path + 'test', data_transform)

    for k_F in tqdm(range(arg.fold)):
        train_loader = torch.utils.data.DataLoader(train_all, batch_size=arg.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)

        # 冻结骨干的 iTPN-Large + 预训练权重
        # Frozen iTPN-Large backbone with pretrained weights
        model = itpn_large(arg.classes)

        opt = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arg.epochs)
        best_test_accuracy = 0

        for epoch in tqdm(range(1, arg.epochs + 1)):
            print("##### " + str(k_F + 1) + " EPOCH " + str(epoch) + "#####")
            loss = model_train(model=model, data_loader=train_loader, opt=opt, sch=scheduler)
            acc = model_test(model, test_loader)
            print('test accuracy is {}'.format(acc))

        history['accuracy'].append(acc)
        print('The best epoch is {}, val accuracy is {}, test accuracy is {}'.format(
            epoch, best_test_accuracy, acc))

    print('OA is {}, STD is {}'.format(np.mean(history['accuracy']), np.std(history['accuracy'])))
    print(history['accuracy'])


if __name__ == '__main__':
    main()
