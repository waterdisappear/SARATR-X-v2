"""
SOC 50 类评测脚本 —— iTPN-Large 变体 (SOC-50 evaluation with iTPN-Large).

在 SOC_50.py 基础上仅将骨干由 itpn_base（512 / depth24）改为 itpn_large
（embed_dim=768, depth=40, num_heads=12, 见 model/models_itpn.py）。

Compared to SOC_50.py, only the backbone is switched from itpn_base
(embed_dim=512, depth=24) to itpn_large (embed_dim=768, depth=40, num_heads=12,
see model/models_itpn.py).

说明：itpn_large() 默认会加载预训练权重（见 resolve_pretrain_ckpt）；也可用 --pretrained
覆盖为其它 large checkpoint。

N-shot：--n_shot N>0 时，训练集按类随机各取至多 N 张（可复现性由 --seed 控制）；
测试集仍全量，并与训练集共用 class_to_idx。

默认训练：backbone 在 no_grad 下前向，池化后 detach，仅 fc_norm+head 参与反向（明显快于整网反传）。
若需整网反传，加 --no_detach_backbone。

仍慢的主要原因：Large 整网前向（训练与 test）仍很重；可用 --eval_every 5、--num_workers 8 减轻。
学习率调度在本脚本中按「每个 epoch 一次」与 CosineAnnealingLR(T_max=epochs) 对齐。
"""
import sys
import os
import argparse
import collections

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from utils.DataLoad import load_data, load_data_with_class_mapping
from utils.TrainTest import model_train, model_val, model_test
from model.models_itpn import itpn_large

# 仓库根目录：用于定位 dataset/ 下数据的默认路径
# (repo root, used to derive the default data path under dataset/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_n_shot_subset(imagefolder_dataset, n_shot: int, seed: int):
    """从 ImageFolder 训练集中每类无放回随机取至多 n_shot 张。

    Randomly sample (without replacement) at most n_shot images per class from
    an ImageFolder training set.
    """
    if n_shot <= 0:
        raise ValueError("n_shot must be positive")

    rng = np.random.RandomState(seed)
    targets = np.array(imagefolder_dataset.targets)
    num_classes = len(imagefolder_dataset.classes)
    chosen = []
    short_classes = []

    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        if len(idx_c) == 0:
            short_classes.append((c, imagefolder_dataset.classes[c], 0))
            continue
        take = min(n_shot, len(idx_c))
        if take < n_shot:
            short_classes.append((c, imagefolder_dataset.classes[c], len(idx_c)))
        pick = rng.choice(idx_c, size=take, replace=False)
        chosen.extend(pick.tolist())

    rng.shuffle(chosen)
    if short_classes:
        print(f"N-shot: {len(short_classes)} class(es) have fewer than {n_shot} images (using all available).")
    print(f"N-shot subset: {n_shot}-shot, total {len(chosen)} train images, seed={seed}.")
    return torch.utils.data.Subset(imagefolder_dataset, chosen)


def parameter_setting():
    parser = argparse.ArgumentParser(description='iTPN-Large on SOC 50 classes')
    default_data = os.path.join(REPO_ROOT, 'dataset', 'classification', 'SOC_50classes') + os.sep
    parser.add_argument('--data_path', type=str, default=default_data,
                        help='data root containing train/ and test/ folders')
    parser.add_argument('--GPU_ids', type=int, default=0,
                        help='GPU ids')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train')
    parser.add_argument('--classes', type=int, default=50,
                        help='number of classes')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size (Large 模型显存占用更大，默认略小于 base 的 128)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--fold', type=int, default=5,
                        help='K-fold')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument(
        '--pretrained',
        type=str,
        default='',
        help='optional MIM/large .pth; expects dict with key model. Empty = default pretrained checkpoint.',
    )
    parser.add_argument(
        '--n_shot',
        type=int,
        default=10,
        help='per-class training shots; 0 = use full train set',
    )
    parser.add_argument(
        '--no_detach_backbone',
        action='store_true',
        help='训练时反向仍穿过整个 backbone（更慢）；默认 backbone 特征 detach，只训 fc_norm+head',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='DataLoader 预取线程数；0 表示主进程读盘（往往更慢）',
    )
    parser.add_argument(
        '--eval_every',
        type=int,
        default=10,
        help='每多少个 epoch 跑一次全量 test；>1 可明显省时间（最后 epoch 仍会测）',
    )
    return parser.parse_args()


def _maybe_load_pretrained(model, path: str):
    """加载额外的预训练权重（可选覆盖默认 checkpoint）。

    Optionally load another pretrained checkpoint on top of the model.
    """
    if not path or not path.strip():
        return
    print(f'Loading pretrained weights from: {path}')
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    for k in ['head.weight', 'head.bias']:
        if k in state and k in model.state_dict():
            if state[k].shape != model.state_dict()[k].shape:
                print(f'Removing mismatched key {k} from checkpoint')
                del state[k]
    msg = model.load_state_dict(state, strict=False)
    print('load_state_dict:', 'missing', len(msg.missing_keys), 'unexpected', len(msg.unexpected_keys))


if __name__ == '__main__':
    arg = parameter_setting()
    torch.cuda.set_device(arg.GPU_ids)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    history = collections.defaultdict(list)

    # 评测数据预处理（SAR 统计值归一化）
    # Evaluation transform (normalization with SAR statistics)
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.2109, 0.2109, 0.2109), (0.2178, 0.2178, 0.2178))
    ])

    # 类别数由训练集自动确定，保证与标签空间一致
    # The number of classes is inferred from the training set to match the label space
    train_all = load_data(arg.data_path + 'train', data_transform)
    global_class_to_idx = train_all.class_to_idx
    n_train_cls = len(train_all.classes)
    if arg.classes != n_train_cls:
        print(f"Warning: --classes={arg.classes} overridden by train folder ({n_train_cls} classes).")
    arg.classes = n_train_cls

    test_set = load_data_with_class_mapping(arg.data_path + 'test', data_transform, global_class_to_idx)

    train_for_loader = train_all
    if arg.n_shot > 0:
        train_for_loader = build_n_shot_subset(train_all, arg.n_shot, arg.seed)

    torch.cuda.set_device(arg.GPU_ids)

    pin = torch.cuda.is_available()
    for k_F in tqdm(range(arg.fold)):
        train_loader = torch.utils.data.DataLoader(
            train_for_loader,
            batch_size=arg.batch_size,
            shuffle=True,
            num_workers=arg.num_workers,
            pin_memory=pin,
            persistent_workers=arg.num_workers > 0,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=arg.batch_size,
            shuffle=False,
            num_workers=arg.num_workers,
            pin_memory=pin,
            persistent_workers=arg.num_workers > 0,
        )

        model = itpn_large(arg.classes)
        _maybe_load_pretrained(model, arg.pretrained)

        # 仅优化仍 requires_grad 的参数（itpn_large 已冻结 backbone）
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=arg.lr,
            weight_decay=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arg.epochs)
        detach_bb = not arg.no_detach_backbone
        if detach_bb:
            print('Training with backbone detached (faster; gradients only on fc_norm + head).')
        best_test_accuracy = 0
        last_acc = 0.0
        for epoch in tqdm(range(1, arg.epochs + 1)):
            print("##### " + str(k_F + 1) + " EPOCH " + str(epoch) + "#####")
            loss = model_train(
                model=model,
                data_loader=train_loader,
                opt=opt,
                sch=scheduler,
                detach_backbone=detach_bb,
                scheduler_step_each_batch=False,
            )
            scheduler.step()
            run_eval = (epoch % arg.eval_every == 0) or (epoch == arg.epochs)
            if run_eval:
                acc = model_test(model, test_loader)
                last_acc = acc
                print(f'Test accuracy: {acc:.2f}%')
            else:
                acc = last_acc
                print(f'Skip test (eval_every={arg.eval_every}); last test acc {last_acc:.2f}%')
        print('test accuracy is {}'.format(acc))
        history['accuracy'].append(acc)
        print('The best epoch is {}, val accuracy is {}, test accuracy is {}'.format(
            epoch, best_test_accuracy, acc))

    print('OA is {}, STD is {}'.format(np.mean(history['accuracy']), np.std(history['accuracy'])))
    print(history['accuracy'])
