"""
HiViT 少样本线性探测（few-shot linear probing with HiViT）

基于 Dassl 框架：冻结预训练 HiViT 骨干，仅训练线性分类头（fc_norm + head）。
用于与 iTPN 线性探测对比的参考实现（SARATR-X v1 使用 HiViT 骨干）。

Based on the Dassl framework: freeze the pretrained HiViT backbone and train
only the linear head. Provided as a reference (the v1 SARATR-X used HiViT).
"""

import os
import os.path as osp
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from model.hivit import HiViT


# HiViT 预训练权重路径 (resolve the HiViT pretrained checkpoint path)：
#   优先使用环境变量 SARATRX_HIVIT_CKPT (env var first)；
#   否则默认指向仓库 weights/base/hivit/checkpoint-1200.pth
#   (otherwise default to weights/base/hivit/checkpoint-1200.pth，见 weights/README.md)。
def resolve_hivit_ckpt() -> str:
    ckpt = os.environ.get("SARATRX_HIVIT_CKPT", "").strip()
    if ckpt:
        return ckpt
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(here, "..", "..", "..", "weights", "base", "hivit", "checkpoint-1200.pth")
    return os.path.normpath(default)


def interpolate_pos_embed(model, checkpoint_model):
    """将 checkpoint 的位置编码插值到当前 patch 网格（尺寸不同时）。

    Interpolate the positional embedding from a checkpoint to the current
    patch grid when the grid sizes differ.
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames):
        super().__init__()
        # HiViT-B 骨干（embed 512, depths=[2,2,20]），加载 MIM 预训练权重（见 resolve_hivit_ckpt）
        # Build the HiViT-B backbone (embed 512, depths=[2,2,20]) and load MIM pretrained weights.
        model = HiViT(
            embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., in_chans=3, mlp_ratio=4.,
            num_classes=len(classnames), ape=True, rpe=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        print(model)

        ckpt_path = resolve_hivit_ckpt()
        print(f"Loading HiViT pretrain checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint = checkpoint['model']

        checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        print('load pre-trained model')
        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # 手动初始化 fc 层（MoCo v3 风格）(manually initialize the fc layer, MoCo v3 style)
        from timm.models.layers import trunc_normal_
        trunc_normal_(model.head.weight, std=0.01)

        # 线性探测：head 前加 BN（affine=False），冻结骨干只训练 fc_norm/head
        # Linear probing: prepend BN (affine=False) to the head, freeze the backbone
        # and train only fc_norm/head.
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
                                         model.head)
        for name, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.fc_norm.named_parameters():
            p.requires_grad = True
        for _, p in model.head.named_parameters():
            p.requires_grad = True

        self.image_encoder = model.cuda()

    def forward(self, image):
        # SAR 单通道复制为 3 通道以匹配 HiViT 输入
        # (duplicate the single-channel SAR image to 3 channels for HiViT input)
        image = torch.concat([image, image, image], 1)
        image_features = self.image_encoder(image)
        return image_features

@TRAINER_REGISTRY.register()
class MIM_linear(TrainerX):
    """ CLIP-Adapter """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading MAE (backbone: {cfg.MODEL.BACKBONE.NAME})')

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames)

        print('Turning off gradients in both the image and the text encoder')
        # for name, param in self.model.named_parameters():
        #     if 'adapter' not in name:
        #         param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            # load_pretrained_weights(self.model.image_encoder, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.image_encoder, cfg.OPTIM)
        # self.optim = build_optimizer(self.model.image_encoder, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model('clip', self.model.image_encoder, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        # loss = F.cross_entropy(output2, label) + self.model.criteria(output1, label)
        loss = F.cross_entropy(output, label)

        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']

            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
        #     curr_result = self.test(split="val")
        #     is_best = curr_result > self.best_result
        #     if is_best:
        #         self.best_result = curr_result
        #         self.save_model(
        #             self.epoch,
        #             self.output_dir,
        #             val_result=curr_result,
        #             model_name="model-best.pth.tar"
        #         )

        # if meet_checkpoint_freq or last_epoch:
        #     self.save_model(self.epoch, self.output_dir)