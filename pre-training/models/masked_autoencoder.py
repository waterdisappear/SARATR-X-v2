# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Reference:
#   MAE : https://github.com/facebookresearch/mae
# --------------------------------------------------------
"""SARATR-X-v2 的“多尺度结构目标”构造与掩码自编码器基类。

对应论文 Method 章节：
  * SAR_Lay  ：最小尺度 S1，3x3 blind-spot 邻域聚合（中心像素置 0），
               保证目标值不包含中心像素自身的相干斑实现，从构造上抑制斑点泄漏；
  * SAR_Layer：大尺度 S2~S6 的 log-ratio 方向对比。将 (2r+1)x(2r+1) 支撑域
               分成两个不相交的半区（中间一行/一列置 0），做对数域均值差：
               g = log(左/上半区) - log(右/下半区)，再对水平/垂直两个方向
               取 L2 范数并过 sigmoid。乘性斑点在同侧按比例抵消，故响应
               主要由“结构不平衡”而非辐射度变化主导；
  * My_SAR_feature：六个尺度分支的输出通过 softmax 约束的可学习权重融合：
               y = sum_s alpha_s * f_s(x),  alpha = softmax(w)
               --target_mode:
                 pixel        : 不使用结构目标，直接回归原始像素；
                 single_s1..6 : 只使用第 s 个单尺度目标（固定权重 one-hot）；
                 multi        : 可学习 softmax 多尺度融合（论文主设置）。

目标提取全程在 torch.no_grad() 下进行，只有融合权重 simple_weights 参与训练，
训练稳定且避免过拟合（对应论文 Implementation Details）。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAR_Lay(nn.Module):
    """S1：最小尺度 blind-spot 聚合分支（3x3 邻域，中心权重为 0）。

    只聚合 8 个相邻像素，使位置 i 的目标值不包含位置 i 自身的斑点实现；
    同时保留边缘、点散射体、纹理边界等局部结构线索。
    """

    def __init__(self, kensize=3):
        super(SAR_Lay, self).__init__()
        self.pi = math.pi
        self.k = kensize
        self.eps = 1e-6

        # 3x3 blind-spot 核：中心为 0，周围 8 个邻居为 1
        weight_LBP1 = torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]])
        weight_LBP2 = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 0.0]]]])

        self.register_buffer("weight_LBP1", weight_LBP1)
        self.register_buffer("weight_LBP2", weight_LBP2)

    @torch.no_grad()
    def forward_a(self, x):
        # 对数域邻域和：log( x * B )，加 eps 防止 log(0)
        gx_1 = torch.log(F.conv2d(x, self.weight_LBP1, bias=None, stride=1, padding=0, groups=1) + self.eps)
        return F.sigmoid(gx_1)

    def forward(self, x):
        # reflect padding 缓解边界伪影；sigmoid 压缩到 (0,1)，保证 log 域稳定
        x = F.sigmoid(F.pad(x, pad=(self.k, self.k, self.k, self.k), mode="reflect"))
        return self.forward_a(x)


class SAR_Layer(nn.Module):
    """S2~S6：大尺度 log-ratio 方向对比分支。

    对 (2k+1)x(2k+1) 的支撑域构造水平/垂直两对“不相交半区”掩码
    （中间一行/一列在两侧掩码中均为 0），计算对数域区域均值之差：
        gx = log(左侧半区) - log(右侧半区)
        gy = log(上侧半区) - log(下侧半区)
    两个方向响应取 L2 范数再 sigmoid：
        f = sigmoid( sqrt(gx^2 + gy^2) )
    乘性斑点对两个半区成比例影响，在对数差中相互抵消，因此该响应
    稳定地反映结构对比而非辐射度变化。
    """

    def __init__(self, kensize=3):
        super(SAR_Layer, self).__init__()
        self.pi = math.pi
        self.k = kensize
        self.eps = 1e-6

        def creat_gauss_kernel(r=1):
            # 两对不相交半区掩码（r 为半宽，M = 2r+1）
            M_13 = np.concatenate([np.ones([r, 2 * r + 1]), np.zeros([r + 1, 2 * r + 1])], axis=0)  # 上半区
            M_23 = np.concatenate([np.zeros([r + 1, 2 * r + 1]), np.ones([r, 2 * r + 1])], axis=0)  # 下半区
            M_11 = np.concatenate([np.ones([2 * r + 1, r]), np.zeros([2 * r + 1, r + 1])], axis=1)  # 左半区
            M_21 = np.concatenate([np.zeros([2 * r + 1, r + 1]), np.ones([2 * r + 1, r])], axis=1)  # 右半区

            return (torch.from_numpy(M_13).float(), torch.from_numpy(M_23).float(),
                    torch.from_numpy(M_11).float(), torch.from_numpy(M_21).float())

        M13, M23, M11, M21 = creat_gauss_kernel(self.k)

        weight_x1 = M11.view(1, 1, self.k * 2 + 1, self.k * 2 + 1)
        weight_x2 = M21.view(1, 1, self.k * 2 + 1, self.k * 2 + 1)
        weight_y1 = M13.view(1, 1, self.k * 2 + 1, self.k * 2 + 1)
        weight_y2 = M23.view(1, 1, self.k * 2 + 1, self.k * 2 + 1)

        self.register_buffer("weight_x1", weight_x1)
        self.register_buffer("weight_x2", weight_x2)
        self.register_buffer("weight_y1", weight_y1)
        self.register_buffer("weight_y2", weight_y2)

    @torch.no_grad()
    def forward_a(self, x):
        # 对数域半区均值
        gx_1 = torch.log(F.conv2d(x, self.weight_x1, bias=None, stride=1, padding=0, groups=1) + self.eps)
        gx_2 = torch.log(F.conv2d(x, self.weight_x2, bias=None, stride=1, padding=0, groups=1) + self.eps)
        gy_1 = torch.log(F.conv2d(x, self.weight_y1, bias=None, stride=1, padding=0, groups=1) + self.eps)
        gy_2 = torch.log(F.conv2d(x, self.weight_y2, bias=None, stride=1, padding=0, groups=1) + self.eps)

        # 水平/垂直方向对比（对数差抵消乘性斑点）
        gx_rgb = gx_1 - gx_2
        gy_rgb = gy_1 - gy_2

        # 两方向响应 L2 聚合后 sigmoid
        norm_rgb1 = F.sigmoid(torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1))
        return norm_rgb1

    @torch.no_grad()
    def forward(self, x):
        x = F.sigmoid(F.pad(x, pad=(self.k, self.k, self.k, self.k), mode="reflect"))
        return self.forward_a(x)


class My_SAR_feature(nn.Module):
    """多尺度结构目标提取器 + 可学习跨尺度融合。

    分支组合（对应论文六个尺度）：
      SAR_Lay(1)   -> S1，3x3 blind-spot 聚合（k=1）
      SAR_Layer(3) -> S2，r=3  log-ratio
      SAR_Layer(5) -> S3，r=5  log-ratio
      SAR_Layer(9) -> S4，r=9  log-ratio
      SAR_Layer(13)-> S5，r=13 log-ratio
      SAR_Layer(17)-> S6，r=17 log-ratio
    即覆盖从 3x3 邻域到 35x35 区域的感受野范围。
    """

    def __init__(self, kensize=3, target_mode: str = "multi"):
        super(My_SAR_feature, self).__init__()
        self.pi = math.pi
        self.k = kensize
        self.target_mode = target_mode

        self.SAR_Lay = SAR_Lay(1)
        self.SAR_Layer0 = SAR_Layer(3)
        self.SAR_Layer1 = SAR_Layer(5)
        self.SAR_Layer2 = SAR_Layer(9)
        self.SAR_Layer3 = SAR_Layer(13)
        self.SAR_Layer4 = SAR_Layer(17)

        # 可学习融合权重（softmax 约束，和为 1），仅 multi 模式下参与训练
        self.simple_weights = nn.Parameter(torch.ones(6))
        self._apply_target_mode(target_mode)

    def _apply_target_mode(self, target_mode: str) -> None:
        """根据 target_mode 配置融合权重。"""
        self.target_mode = target_mode
        if target_mode == "multi":
            # 论文主设置：softmax 可学习跨尺度融合
            self.simple_weights.requires_grad = True
            with torch.no_grad():
                self.simple_weights.fill_(1.0)
            return
        if target_mode.startswith("single_s"):
            # 固定单尺度目标：对应尺度的权重为 1，其余为 0（不参与训练）
            idx = int(target_mode.replace("single_s", "")) - 1
            if not (0 <= idx < 6):
                raise ValueError(f"Invalid target_mode: {target_mode}")
            with torch.no_grad():
                self.simple_weights.zero_()
                self.simple_weights[idx] = 1.0
            self.simple_weights.requires_grad = False
            return
        if target_mode == "pixel":
            # 像素重建：不使用结构目标，权重保留但无意义
            self.simple_weights.requires_grad = False
            return
        raise ValueError(f"Unknown target_mode: {target_mode}")

    def configure_target(self, target_mode: str) -> None:
        self._apply_target_mode(target_mode)

    def forward(self, x):
        # 目标提取全程 no_grad：只有融合权重参与训练，训练稳定
        with torch.no_grad():
            y1 = self.SAR_Lay(x)     # S1 blind-spot
            y2 = self.SAR_Layer0(x)  # S2 r=3
            y3 = self.SAR_Layer1(x)  # S3 r=5
            y4 = self.SAR_Layer2(x)  # S4 r=9
            y5 = self.SAR_Layer3(x)  # S5 r=13
            y6 = self.SAR_Layer4(x)  # S6 r=17

        if self.target_mode.startswith("single_s"):
            idx = int(self.target_mode.replace("single_s", "")) - 1
            return [y1, y2, y3, y4, y5, y6][idx]

        # 多尺度 softmax 融合：y = sum_s alpha_s * f_s(x)
        weights = F.softmax(self.simple_weights, dim=0)
        fused_feature = (
            weights[0] * y1
            + weights[1] * y2
            + weights[2] * y3
            + weights[3] * y4
            + weights[4] * y5
            + weights[5] * y6
        )
        return fused_feature


class MaskedAutoencoder(nn.Module):
    """掩码自编码器基类：负责 patchify / 目标构造 / 损失计算。

    forward 流程：
      1. 单通道 SAR 图复制成 3 通道送入编码器（兼容 patch embedding）；
      2. 编码器得到多尺度 latent -> decoder 恢复掩码位置；
      3. forward_loss 在掩码 patch 上计算 L2 损失；
         目标 = pixel（像素重建）或 hogs(x)（多尺度结构目标），
         并做逐 patch 归一化（norm_pix_loss）。
    """

    def __init__(self, target_mode: str = "multi"):
        nn.Module.__init__(self)
        self.norm_pix_loss = True
        self.target_mode = target_mode
        self.hogs = My_SAR_feature(kensize=1, target_mode=target_mode)

    def configure_target(self, target_mode: str) -> None:
        self.target_mode = target_mode
        self.hogs.configure_target(target_mode)

    def patchify(self, imgs):
        """imgs: (N, C, H, W) -> x: (N, L, patch_size**2 * C)"""
        p = self.decoder_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * imgs.shape[1]))
        return x

    def unpatchify(self, x):
        """x: (N, L, patch_size**2 * C) -> imgs: (N, C, H, W)"""
        p = self.decoder_patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def masking_id(self, batch_size, mask_ratio):
        N, L = batch_size, self.patch_embed.num_patches
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=self.pos_embed.device)  # noise in [0, 1]

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([N, L], device=self.pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def random_masking(self, x, ids_keep):
        N, L, D = x.shape
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward_encoder(self, x, mask_ratio):
        raise NotImplementedError

    def forward_decoder(self, x, ids_restore):
        raise NotImplementedError

    def forward_loss(self, imgs, cls_pred, pred, mask):
        """在掩码 patch 上计算 L2 重建损失。

        imgs: [N, 1, H, W]（单通道 SAR 图）
        pred: [N, L, p*p*1]
        mask: [N, L]，0 保留、1 掩码
        """
        num_preds = mask.sum()
        if self.target_mode == "pixel":
            target = self.patchify(imgs)
        else:
            y = self.hogs(imgs)          # 多尺度结构目标（融合后仍为单通道）
            target = self.patchify(y)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds
        return loss

    def forward(self, img, mask_ratio=0.75):
        imgs = torch.cat([img, img, img], dim=1)   # 单通道 -> 3 通道
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        cls_pred, pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(img, cls_pred, pred, mask)
        return loss, pred, mask
