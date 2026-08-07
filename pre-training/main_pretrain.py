# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
#   iTPN : https://github.com/sunny2109/iTPN
#   MAE  : https://github.com/facebookresearch/mae
#   BEiT : https://github.com/microsoft/unilm/tree/master/beit
#   HiViT: https://github.com/zhangxiaosong18/hivit
# --------------------------------------------------------
"""SARATR-X-v2 预训练主入口。

在 500K 张无标注 SAR 目标切片上进行“尺度感知结构掩码预训练”（Scale-Aware
Structural Pre-training）。核心思想见论文 Method 章节：

  * 编码器：iTPN 分层 Transformer + FPN 金字塔，输出多尺度 latent。
  * 目标  ：从输入 SAR 图构造“多尺度结构目标 y”：
      - S1: 3x3 blind-spot 邻域聚合（中心像素置 0，抑制相干斑泄漏）；
      - S2~S6: 五组大尺度 log-ratio 方向对比（r = 3/5/9/13/17）；
      - 融合: softmax 约束的可学习跨尺度权重 y = sum_s alpha_s * f_s(x)。
  * 损失  ：仅在掩码 patch 上计算预测目标与真实目标的 L2 距离。

--target_mode 支持 8 种目标设置（论文消融）：
  pixel       : 原始像素重建（基线）
  single_s1..6: 固定单尺度目标
  multi       : 可学习 softmax 多尺度融合（默认，论文主设置）
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import load_data

import models
from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('iTPN masked pre-training (SARATR-X-v2)', add_help=False)
    # ------------------------------------------------------------------
    # 训练基本参数
    # ------------------------------------------------------------------
    parser.add_argument('--batch_size', default=200, type=int,
                        help='单卡 batch size（有效 batch = batch_size * accum_iter * GPU 数）')
    parser.add_argument('--epochs', default=1200, type=int, help='训练轮数（论文主实验为 1200）')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='梯度累积步数，用于在显存受限时扩大有效 batch size')
    parser.add_argument('--input_size', default=224, type=int, help='输入图像尺寸')

    # ------------------------------------------------------------------
    # 模型参数
    # ------------------------------------------------------------------
    parser.add_argument('--model', default='itpn_base_dec512d8b', type=str, metavar='MODEL',
                        help='模型名称：itpn_base_dec512d8b / itpn_large_dec512d8b')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='掩码比例（被移除 patch 的占比）')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='对目标 patch 做逐 patch 归一化后再算损失（默认开启）')
    parser.set_defaults(norm_pix_loss=True)

    # 重建目标模式（v2 核心消融项）
    parser.add_argument(
        '--target_mode', default='multi', type=str,
        choices=('pixel', 'multi', 'single_s1', 'single_s2', 'single_s3',
                 'single_s4', 'single_s5', 'single_s6'),
        help='重建目标：pixel=像素重建 | single_s1..s6=固定单尺度结构目标 | multi=多尺度 softmax 融合',
    )

    # 预训练初始化权重（可选）。默认留空；若提供 iTPN ImageNet 权重
    # （例如官方 iTPN 的 itpn_base_fpn256.pth），会做 warm-start。
    parser.add_argument('--init_ckpt', default='', type=str,
                        help='warm-start 用预训练权重路径（如 itpn_base_fpn256.pth），可留空')

    # ------------------------------------------------------------------
    # 优化器参数
    # ------------------------------------------------------------------
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='绝对学习率（默认由 blr 自动计算）')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='基础学习率：absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='循环学习率下界')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='warmup 轮数')

    # ------------------------------------------------------------------
    # 数据参数
    # ------------------------------------------------------------------
    parser.add_argument('--data_path', default='./data/500K', type=str,
                        help='预训练数据路径。默认按子文件夹（ImageFolder 风格）读取 SAR 图片，'
                             '如需按文件列表读取，请改用 load_data / SARImageDataset（见 util/datasets.py）')
    parser.add_argument('--output_dir', default='./output',
                        help='模型保存路径，留空则不保存')
    parser.add_argument('--log_dir', default=None,
                        help='TensorBoard 日志路径')
    parser.add_argument('--device', default='cuda', help='训练设备')
    parser.add_argument('--seed', default=0, type=int, help='随机种子')
    parser.add_argument('--resume', default='', help='从 checkpoint 恢复训练')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='起始轮数')
    parser.add_argument('--num_workers', default=8, type=int, help='DataLoader worker 数')
    parser.add_argument('--pin_mem', action='store_true',
                        help='将样本固定到 CPU 内存，提升搬运到 GPU 的效率')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # ------------------------------------------------------------------
    # 分布式训练参数
    # ------------------------------------------------------------------
    parser.add_argument('--world_size', default=1, type=int, help='分布式进程总数')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='分布式 init url')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()

    if args.log_dir is None:
        args.log_dir = args.output_dir

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 固定随机种子，保证可复现
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # ------------------------------------------------------------------
    # 数据增强（针对 SAR 幅度图，参照 SARATR-X）：
    #   随机裁剪 + 水平翻转 + 轻微对比度扰动，配合 SAR 专用归一化统计量
    #   （mu=0.2109, std=0.2178，由 10000 张随机 SAR 幅度图统计得到）
    # ------------------------------------------------------------------
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 = bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2109], std=[0.2178]),
    ])

    # 数据加载：默认 ImageFolder 风格；也可改用 SARImageDataset 按文件列表加载
    dataset_train = load_data(os.path.join(args.data_path), transform=transform_train)
    print(f"Train dataset size: {len(dataset_train)}")

    num_tasks = misc.get_world_size()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # 构建模型并配置重建目标模式
    # ------------------------------------------------------------------
    model = models.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.norm_pix_loss = args.norm_pix_loss
    if hasattr(model, "configure_target"):
        model.configure_target(args.target_mode)
    print(f"target_mode = {args.target_mode}")

    # 可选：加载 ImageNet/iTPN 预训练权重做 warm-start（非严格匹配）
    if args.init_ckpt:
        checkpoint = torch.load(args.init_ckpt, map_location='cpu')
        if 'model' in checkpoint:            # 兼容包含 'model' 键的 checkpoint
            checkpoint = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['decoder_pred.weight', 'decoder_pred.bias']:
            if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        msg = model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded init checkpoint from {args.init_ckpt}:")
        print(msg)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # 只给 blr 时按比例换算绝对学习率
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # 参照 timm：bias 与 norm 层不衰减
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    # 自动恢复 output_dir 中最新的 checkpoint
    if len(args.resume) == 0:
        try:
            last_epoch = -1
            for ckpt in os.listdir(args.output_dir):
                if ckpt[-4:] == '.pth':
                    epoch = int(ckpt[:-4].split('-')[1])
                    last_epoch = max(last_epoch, epoch)
            if last_epoch >= 0:
                args.resume = f'{args.output_dir}/checkpoint-{last_epoch}.pth'
        except Exception:
            pass

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        # 每 50 轮与最后一轮保存一次 checkpoint
        if args.output_dir and (epoch + 1 == args.epochs or (epoch > 0 and epoch % 50 == 0)):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
