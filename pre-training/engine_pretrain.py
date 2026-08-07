# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
"""预训练单轮训练函数：混合精度 + 梯度累积 + 梯度裁剪 + LR 调度。"""

import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # 在循环开始前初始化 total_norm
    total_norm = 0.0

    # 在开始循环前初始化指标
    metric_logger.update(loss=0.0, lr=optimizer.param_groups[0]["lr"], grad_norm=total_norm)

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        # 更严格的NaN检查
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            # 保存当前状态以进行调试
            torch.save({
                'epoch': epoch,
                'step': data_iter_step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'samples': samples.cpu(),
            }, 'debug_nan.pth')
            sys.exit(1)

        loss /= accum_iter

        # 判断是否是梯度累积的最后一步
        is_update_step = (data_iter_step + 1) % accum_iter == 0
        
        # 在 loss_scaler 调用后处理梯度（关键修复）
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=is_update_step,
                    clip_grad=1.0 if is_update_step else None)  # 在scaler内部进行梯度裁剪

        # 如果是更新步骤，计算梯度范数
        if is_update_step:
            # 计算梯度范数
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5 if total_norm > 0 else 0.0
            
            metric_logger.update(grad_norm=total_norm)
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and data_iter_step == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            # 确保 total_norm 已经定义
            log_writer.add_scalar('grad_norm', total_norm, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}