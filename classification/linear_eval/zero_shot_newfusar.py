"""
FUSAR-Ship k-NN 评测（Zero-shot k-NN evaluation on FUSAR-Ship）

论文中 FUSAR-Ship 使用冻结 backbone 的 k-NN 分类评测（高分辨率 384x384）。
在 zero_shot.py 基础上，仅将分辨率改为 384 并默认指向 FUSAR-Ship 数据。

For FUSAR-Ship the paper evaluates frozen features with k-NN classification.
Compared to zero_shot.py, the input resolution is 384 and the default data
root points to the FUSAR-Ship dataset.

用法 (Usage)::

    python zero_shot_newfusar.py --mode knn --backbone base --k_values 1,5,10,20
"""

import os
import argparse
import collections

import numpy as np
import torch
import torchvision

from utils.DataLoad import load_data, load_data_with_class_mapping
from utils.TrainTest import model_train, model_test
from utils.knn_eval import (build_itpn, load_pretrained_itpn, resolve_pretrain_ckpt,
                            build_knn_train_subset, zero_shot_evaluation)

# 仓库根目录：用于定位 dataset/ 下数据的默认路径
# (repo root, used to derive the default data path under dataset/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parameter_setting():
    parser = argparse.ArgumentParser(description='iTPN k-NN evaluation on FUSAR-Ship')
    default_data = os.path.join(REPO_ROOT, 'dataset', 'classification', 'FUSAR-Ship') + os.sep
    parser.add_argument('--data_path', type=str, default=default_data,
                        help='data root containing Train/ and Val/ folders')
    parser.add_argument('--GPU_ids', type=int, default=0, help='GPU id')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs for fine-tuning (if enabled)')
    parser.add_argument('--classes', type=int, default=0,
                        help='number of classes (0=auto from Train ImageFolder)')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for fine-tuning')
    parser.add_argument('--fold', type=int, default=5,
                        help='k-NN: 不同随机训练子集重复次数；finetune: K 次独立训练')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='knn', choices=['knn', 'finetune'],
                        help='evaluation mode: knn (frozen features) or finetune (linear head)')
    parser.add_argument('--n_shot', type=int, default=0,
                        help='per-class shots from Train; 0=use full Train')
    parser.add_argument('--backbone', type=str, default='base', choices=['base', 'large'],
                        help='iTPN structure: base (512/24) or large (768/40)')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='explicit pretrained .pth; empty = resolve automatically')
    parser.add_argument('--k_values', type=str, default='1,5,10,20',
                        help='k values for k-NN (comma separated)')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='temperature for softmax in k-NN')
    parser.add_argument('--feature_layer', type=str, default='gap',
                        choices=['gap', 'pooled', 'fc_norm'],
                        help='k-NN features: gap/pooled=GAP only; fc_norm=also apply fc_norm')
    parser.add_argument('--skip_first_nn', action='store_true',
                        help='skip first nearest neighbor')
    return parser.parse_args()


def main():
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)

    # 高分辨率数据预处理：FUSAR-Ship 采用 384x384
    # High-resolution transform: FUSAR-Ship uses 384x384 input
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384),
        torchvision.transforms.CenterCrop(384),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.2109, 0.2109, 0.2109), (0.2178, 0.2178, 0.2178))
    ])

    # 加载数据，类别数由训练集自动确定
    # Load data; the number of classes is inferred from the training set
    train_all = load_data(arg.data_path + 'Train', data_transform)
    global_class_to_idx = train_all.class_to_idx
    num_train_classes = len(train_all.classes)
    if arg.classes and arg.classes != num_train_classes:
        print(f"Warning: --classes={arg.classes} overridden by Train folder ({num_train_classes} classes).")
    arg.classes = num_train_classes

    test_set = load_data_with_class_mapping(arg.data_path + 'Val', data_transform, global_class_to_idx)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=arg.batch_size, shuffle=False)

    # 构造 iTPN 骨干并加载预训练权重（见 utils/knn_eval.py）
    # Build the iTPN backbone and load pretrained weights (see utils/knn_eval.py)
    model = build_itpn(arg.backbone, num_classes=arg.classes)
    ckpt_path = arg.checkpoint.strip() if arg.checkpoint.strip() else resolve_pretrain_ckpt(arg.backbone)
    load_pretrained_itpn(model, ckpt_path)

    if arg.mode == 'knn':
        # k-NN 评测：冻结全部参数，多折取均值 ± 标准差
        # k-NN evaluation: freeze all parameters, report mean ± std over folds
        print("=" * 50)
        print(f"Zero-shot k-NN Classification on FUSAR-Ship (backbone={arg.backbone})")
        print(f"Folds: {arg.fold} (train subset seed = --seed + fold_index)")
        print("=" * 50)

        for param in model.parameters():
            param.requires_grad = False

        history_knn = collections.defaultdict(list)
        for k_F in range(arg.fold):
            train_subset = build_knn_train_subset(train_all, arg, k_F)
            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=arg.batch_size, shuffle=True
            )
            print(f"\n----- k-NN fold {k_F + 1}/{arg.fold} | train images: {len(train_subset)} -----")
            results = zero_shot_evaluation(model, train_loader, test_loader, arg)
            for metric, value in results.items():
                history_knn[metric].append(value)
                print(f"  {metric}: {value:.2f}%")

        print("\n" + "=" * 50)
        print("Zero-shot k-NN Results (mean ± std over folds):")
        for metric in sorted(history_knn.keys()):
            vals = history_knn[metric]
            print(f"{metric}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%  (all: {[round(v, 2) for v in vals]})")
        print("=" * 50)

    elif arg.mode == 'finetune':
        # 线性微调模式：冻结骨干，只训练 fc_norm + head
        # Linear fine-tuning mode: freeze backbone, train fc_norm + head only
        print("=" * 50)
        print("Fine-tuning Mode")
        print("=" * 50)

        train_loader = torch.utils.data.DataLoader(
            build_knn_train_subset(train_all, arg, 0),
            batch_size=arg.batch_size, shuffle=True,
        )

        from timm.models.layers import trunc_normal_
        trunc_normal_(model.head.weight, std=0.01)
        model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
            model.head
        )
        for name, param in model.named_parameters():
            param.requires_grad = False
        for _, param in model.fc_norm.named_parameters():
            param.requires_grad = True
        for _, param in model.head.named_parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=arg.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=arg.epochs)

        history = collections.defaultdict(list)
        for k_F in range(arg.fold):
            best_test_accuracy = 0
            for epoch in range(1, arg.epochs + 1):
                print(f"##### Fold {k_F + 1} Epoch {epoch} #####")
                model_train(model, train_loader, optimizer, scheduler)
                acc = model_test(model, test_loader)
                print(f'Test accuracy: {acc:.2f}%')
                if acc > best_test_accuracy:
                    best_test_accuracy = acc
            history['accuracy'].append(best_test_accuracy)
            print(f'Fold {k_F + 1} - Best test accuracy: {best_test_accuracy:.2f}%')

        print('Overall Accuracy: {:.2f}%, STD: {:.2f}'.format(
            np.mean(history['accuracy']), np.std(history['accuracy'])))


if __name__ == '__main__':
    main()
