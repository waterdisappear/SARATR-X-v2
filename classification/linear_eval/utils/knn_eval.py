"""
k-NN 评测公共工具 (Shared utilities for k-NN evaluation)

论文中 SAR-VSA / FUSAR-Ship 使用冻结特征的 k-NN 分类评测。
本模块提供：
  - build_itpn            : 构造 iTPN-Base / iTPN-Large 骨干
  - load_pretrained_itpn  : 加载 MIM 预训练权重（去除分类头）
  - build_n_shot_subset   : 每类取至多 N 张构成 N-shot 子集
  - build_knn_train_subset: 构造 k-NN 训练子集（N-shot 或 30% 随机划分）
  - extract_features      : 提取 L2 归一化特征
  - knn_classification    : faiss k-NN 分类（温度缩放加权投票）
  - zero_shot_evaluation  : 完整 k-NN 评测流程

For SAR-VSA / FUSAR-Ship the paper evaluates frozen features with k-NN classification.
"""

import os
from functools import partial

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.models_itpn import iTPN

# 仓库根目录（用于定位 weights/ 默认权重路径）
# (repo root, used to locate the default weights under weights/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def resolve_pretrain_ckpt(backbone: str = 'base') -> str:
    """解析预训练权重路径 (Resolve the pretrained checkpoint path).

    优先级 (priority):
      1. 环境变量 SARATRX_PRETRAIN_CKPT（base）/ SARATRX_PRETRAIN_CKPT_LARGE（large）
         (environment variables SARATRX_PRETRAIN_CKPT / SARATRX_PRETRAIN_CKPT_LARGE)
      2. 仓库 weights/ 目录下的默认权重 (default weights under weights/)
    """
    env_key = 'SARATRX_PRETRAIN_CKPT' if backbone == 'base' else 'SARATRX_PRETRAIN_CKPT_LARGE'
    ckpt = os.environ.get(env_key, '').strip()
    if ckpt:
        return ckpt
    if backbone == 'base':
        default = os.path.join(REPO_ROOT, 'weights', 'base', 'jiaquan_simple', 'checkpoint-1200.pth')
    else:
        default = os.path.join(REPO_ROOT, 'weights', 'large', 'jiaquan_simple', 'checkpoint-1200.pth')
    return os.path.normpath(default)


def build_itpn(backbone: str = 'base', num_classes: int = 0) -> iTPN:
    """构造 iTPN 骨干 (Build an iTPN backbone).

    Args:
        backbone: 'base'（embed 512 / depth 24）或 'large'（embed 768 / depth 40）
        num_classes: 分类头维度（k-NN 评测中通常为 0，即不训练分类头）
    """
    if backbone == 'base':
        return iTPN(
            embed_dim=512, mlp_depth=3, depth=24, num_heads=8,
            bridge_mlp_ratio=3., mlp_ratio=4., num_classes=num_classes,
            rpe=False, num_outs=1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
    return iTPN(
        embed_dim=768, mlp_depth=2, depth=40, num_heads=12,
        bridge_mlp_ratio=3., mlp_ratio=4., num_classes=num_classes,
        rpe=False, num_outs=1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def interpolate_pos_embed(model, checkpoint_model):
    """将 checkpoint 中的位置编码插值到当前 patch 网格。

    Interpolate the positional embedding from a checkpoint to the current patch grid.
    iTPN 中参数名为 absolute_pos_embed；部分权重仍保存为 pos_embed。
    """
    key = None
    for k in ('pos_embed', 'absolute_pos_embed'):
        if k in checkpoint_model:
            key = k
            break
    if key is None:
        return

    pos = checkpoint_model[key]
    if pos.dim() != 3:
        return
    _, seq_len, emb_dim = pos.shape
    num_patches = model.patch_embed.num_patches
    new_side = int(num_patches ** 0.5)

    num_extra = None
    orig_side = None
    for n_extra in (0, 1, 2):
        n_patch_tok = seq_len - n_extra
        if n_patch_tok <= 0:
            continue
        s = int(round(n_patch_tok ** 0.5))
        if s * s == n_patch_tok:
            num_extra = n_extra
            orig_side = s
            break
    if num_extra is None or orig_side is None:
        print(f"Warning: cannot parse pos embed (seq_len={seq_len}); skip interpolate.")
        return

    if orig_side == new_side and (seq_len - num_extra) == num_patches:
        if key != 'absolute_pos_embed':
            checkpoint_model['absolute_pos_embed'] = checkpoint_model.pop(key)
        return

    print(f"Position interpolate from {orig_side}x{orig_side} to {new_side}x{new_side}")
    extra = pos[:, :num_extra] if num_extra else pos[:, :0]
    pos_tokens = pos[:, num_extra:]
    pos_tokens = pos_tokens.reshape(1, orig_side, orig_side, emb_dim).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_side, new_side), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, -1, emb_dim)
    new_pe = torch.cat((extra, pos_tokens), dim=1) if num_extra else pos_tokens

    if key in checkpoint_model and key != 'absolute_pos_embed':
        del checkpoint_model[key]
    checkpoint_model['absolute_pos_embed'] = new_pe


def load_pretrained_itpn(model, ckpt_path: str):
    """从 MIM 预训练权重加载 iTPN（移除分类头 head.weight / head.bias）。

    Load iTPN weights from a MIM pretrained checkpoint (removing the classification head).
    """
    print(f"Loading pretrained checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint = checkpoint['model']
    checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)


def build_n_shot_subset(imagefolder_dataset, n_shot: int, seed: int):
    """从 ImageFolder 训练集中按类各取至多 n_shot 张，构成 N-shot 训练子集。

    Build an N-shot subset by sampling at most n_shot images per class from an
    ImageFolder training set (deterministic given the seed).
    """
    if n_shot <= 0:
        raise ValueError("n_shot must be positive when building N-shot subset")

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
        print(f"N-shot: {len(short_classes)} class(es) have fewer than {n_shot} samples (using all available).")
    print(f"N-shot subset: {n_shot}-shot, total {len(chosen)} images, seed={seed}.")
    return torch.utils.data.Subset(imagefolder_dataset, chosen)


def build_knn_train_subset(train_all, args, fold_idx: int):
    """构造 k-NN / 训练用子集：n_shot>0 时为每类至多 n_shot 张；否则取训练集 30% 随机划分。

    Build the training subset for k-NN: per-class N-shot subset if n_shot>0,
    otherwise a 30% random split of the training set.
    """
    rng_seed = int(args.seed) + int(fold_idx)
    if getattr(args, 'n_shot', 0) > 0:
        return build_n_shot_subset(train_all, args.n_shot, rng_seed)
    total_train = len(train_all)
    train_size = int(total_train * 0.3)
    remain_size = total_train - train_size
    from torch.utils.data import random_split
    train_subset, _ = random_split(
        train_all,
        [train_size, remain_size],
        generator=torch.Generator().manual_seed(rng_seed),
    )
    return train_subset


def extract_features(model, data_loader, device, feature_layer='gap'):
    """提取 L2 归一化的特征 (Extract L2-normalized features).

    feature_layer:
      - gap / pooled : backbone 全局平均池化后直接 L2（推荐，匹配无 fc_norm 的预训练权重）
                       (GAP then L2; recommended for backbones without a trained fc_norm)
      - fc_norm      : GAP 后再过 fc_norm 再 L2（仅当权重来自已训 fc_norm 的 checkpoint）
                       (GAP, then fc_norm, then L2)
    """
    model.eval()
    all_features = []
    all_labels = []
    fl = (feature_layer or 'gap').lower().strip()
    use_fc_norm = fl == 'fc_norm'
    if fl not in ('gap', 'pooled', 'fc_norm'):
        raise ValueError(f"feature_layer must be gap, pooled, or fc_norm, got {feature_layer!r}")

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)

            features = model.forward_features(images)
            if isinstance(features, (list, tuple)):
                features = features[0]

            # 全局平均池化 (global average pooling)
            features = features.mean(dim=1)

            if use_fc_norm and hasattr(model, 'fc_norm'):
                features = model.fc_norm(features)

            # L2 归一化 (L2 normalization)
            features = F.normalize(features, dim=-1)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def knn_classification(train_features, train_labels, test_features, test_labels,
                       k_values=(1, 5, 10, 20), temperature=0.07, skip_first_nn=False,
                       num_classes_override=None):
    """基于 faiss 的 k-NN 分类（温度缩放加权投票）。

    faiss-based k-NN classification with temperature-scaled weighted voting.
    """
    device = train_features.device
    if num_classes_override is not None:
        num_classes = int(num_classes_override)
    else:
        num_classes = int(train_labels.max().item()) + 1

    train_features_np = train_features.numpy().astype('float32')
    test_features_np = test_features.numpy().astype('float32')

    faiss.normalize_L2(train_features_np)
    faiss.normalize_L2(test_features_np)

    d = train_features_np.shape[1]
    index = faiss.IndexFlatIP(d)  # 内积 = 余弦相似度 (inner product = cosine similarity)
    index.add(train_features_np)

    max_k = max(k_values) + (1 if skip_first_nn else 0)
    D, I = index.search(test_features_np, max_k)  # D: 相似度, I: 近邻索引

    if skip_first_nn:
        D = D[:, 1:]
        I = I[:, 1:]
        max_k -= 1

    D = torch.from_numpy(D).to(device)
    I = torch.from_numpy(I).to(device)
    train_labels = train_labels.to(device)
    test_labels = test_labels.to(device)

    results = {}
    for k in k_values:
        k = min(k, max_k)

        neighbors_labels = train_labels[I[:, :k]]
        weights = F.softmax(D[:, :k] / temperature, dim=1)  # 温度缩放加权 (temperature-scaled weights)

        batch_size = test_features.shape[0]
        scores = torch.zeros(batch_size, num_classes, device=device)
        for i in range(batch_size):
            for j in range(k):
                label = neighbors_labels[i, j]
                scores[i, label] += weights[i, j]

        predictions = torch.argmax(scores, dim=1)
        accuracy = (predictions == test_labels).float().mean().item() * 100
        results[f'Top-1 (k={k})'] = accuracy

        if num_classes >= 5:
            _, top5_pred = scores.topk(5, dim=1)
            top5_correct = torch.any(top5_pred == test_labels.unsqueeze(1), dim=1)
            top5_accuracy = top5_correct.float().mean().item() * 100
            results[f'Top-5 (k={k})'] = top5_accuracy

    return results


def zero_shot_evaluation(model, train_loader, test_loader, args):
    """完整 k-NN 评测流程 (Full k-NN evaluation pipeline)."""
    device = torch.device(f'cuda:{args.GPU_ids}' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Extracting features from training set...")
    train_features, train_labels = extract_features(model, train_loader, device, args.feature_layer)

    print("Extracting features from test set...")
    test_features, test_labels = extract_features(model, test_loader, device, args.feature_layer)

    k_values = [int(k) for k in args.k_values.split(',')]
    print(f"Performing k-NN classification with k={k_values}...")

    results = knn_classification(
        train_features, train_labels,
        test_features, test_labels,
        k_values=k_values,
        temperature=args.temperature,
        skip_first_nn=args.skip_first_nn,
        num_classes_override=getattr(args, 'classes', None),
    )
    return results
