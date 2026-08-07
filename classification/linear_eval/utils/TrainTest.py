"""
训练与评测工具 (Training & evaluation utilities)

线性评测使用的训练 / 测试函数：
  - model_train : 训练一个 epoch（可选 backbone 特征 detach，只反向更新分类头）
                  (Train one epoch; optionally detach backbone features so that only the head is updated)
  - model_test  : 在测试集上评估 Top-1 准确率
                  (Evaluate Top-1 accuracy on the test set)
  - model_val   : 同 model_test，用于验证
                  (Same as model_test, used for validation)
"""

import numpy as np
import torch
import torch.nn as nn


def _forward_itpn_head_detached(model, x):
    """在 no_grad 下完成 backbone 前向并 detach 特征，反向只更新 fc_norm / head。

    Run the backbone forward under no_grad and detach features, so gradients only
    update fc_norm / head. This is the default fast path used by iTPN linear probing.
    """
    with torch.no_grad():
        feat = model.forward_features(x)
        if isinstance(feat, (list, tuple)):
            feat = feat[0] if isinstance(feat, list) else feat
        feat = feat.mean(dim=1)
    z = feat.detach()
    z = model.fc_norm(z)
    return model.head(z)


def model_train(model, data_loader, opt, sch, detach_backbone=False, scheduler_step_each_batch=True):
    """训练一个 epoch (Train the model for one epoch).

    Args:
        model: 待训练模型
        data_loader: 训练数据加载器
        opt: 优化器
        sch: 学习率调度器（可为 None）
        detach_backbone: 为 True 时 backbone 前向在 no_grad 下完成（仅训练分类头）
        scheduler_step_each_batch: 每个 batch 是否 step 一次调度器
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    correct = 0
    cr1 = nn.CrossEntropyLoss()

    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        if detach_backbone and hasattr(model, 'forward_features') and hasattr(model, 'fc_norm'):
            output = _forward_itpn_head_detached(model, x)
        else:
            output = model(x)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

        loss = cr1(output, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch is not None and scheduler_step_each_batch:
            sch.step()

    print("Train Accuracy is:{:.2f} %: ".format(100. * correct / len(data_loader.dataset)))
    return loss.item()


def _evaluate(model, data_loader):
    """在给定加载器上计算 Top-1 准确率（无梯度）。

    Compute Top-1 accuracy on the given loader (no gradient).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(data_loader.dataset)


def model_test(model, test_loader):
    """在测试集上评估 Top-1 准确率 (Evaluate Top-1 accuracy on the test set)."""
    return _evaluate(model, test_loader)


def model_val(model, val_loader):
    """在验证集上评估 Top-1 准确率 (Evaluate Top-1 accuracy on the validation set)."""
    return _evaluate(model, val_loader)
