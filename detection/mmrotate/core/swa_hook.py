# Copyright (c) OpenMMLab. All rights reserved.
"""SWA (Stochastic Weight Averaging) hook for mmcv EpochBasedRunner.

mmdet 的 EvalHook 在 ``after_train_epoch`` 内直接跑 ``single_gpu_test``，不会走
``before_val_epoch``。因此：在本 epoch 的 ``after_train_epoch`` 里先 ``update_parameters``，
再备份并载入 SWA 权重，让随后同阶段、更大 priority 的 EvalHook 用 SWA 模型评测；
下一 epoch 的 ``before_train_epoch`` 再恢复训练权重。

训练结束保存 ``work_dir/swa_model.pth``。
"""
import os.path as osp

from mmcv.runner import HOOKS, Hook

try:
    from mmcv.runner.checkpoint import save_checkpoint
except ImportError:  # pragma: no cover
    from mmcv.runner import save_checkpoint

try:
    from torch.optim.swa_utils import AveragedModel
except ImportError:  # pragma: no cover
    AveragedModel = None


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class SWAHook(Hook):
    """SWA：从第 ``swa_start`` 个 epoch（与日志 Epoch [k] 的 k 一致）起更新平均模型。

    Args:
        swa_start (int): 从该 epoch 起开始 SWA（1-based，与训练 log 中 Epoch [24] 一致）。
        eval_with_swa (bool): 若为 True，SWA 阶段每次 val 前载入平均权重（见模块说明）。
        save_at_end (bool): 训练结束时是否保存 ``work_dir/swa_model.pth``。

    Note:
        执行顺序由 **配置里的** ``priority`` 控制（勿在类里写 ``self.priority``，mmcv
        ``register_hook`` 会报错 ``priority is a reserved attribute``）。须小于 EvalHook(50)。
    """

    def __init__(self,
                 swa_start=24,
                 eval_with_swa=True,
                 save_at_end=True):
        assert AveragedModel is not None, (
            'SWAHook requires PyTorch with torch.optim.swa_utils.AveragedModel')
        self.swa_start = int(swa_start)
        self.eval_with_swa = eval_with_swa
        self.save_at_end = save_at_end
        self.avg_model = None

    def before_run(self, runner):
        model = _unwrap(runner.model)
        self.avg_model = AveragedModel(model)

    def after_train_epoch(self, runner):
        """与日志 Epoch [N] 对齐：runner.epoch 为刚完成的 epoch 的 0-based 索引。"""
        done_display_epoch = runner.epoch + 1
        if done_display_epoch >= self.swa_start:
            self.avg_model.update_parameters(_unwrap(runner.model))

        if (not self.eval_with_swa or done_display_epoch < self.swa_start
                or self.avg_model is None):
            runner._swa_train_backup = None
            return

        m = _unwrap(runner.model)
        runner._swa_train_backup = {
            k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        device = next(m.parameters()).device
        m.load_state_dict(
            {k: v.to(device) for k, v in self.avg_model.module.state_dict().items()})

    def after_run(self, runner):
        if not self.save_at_end or self.avg_model is None:
            return
        rank = getattr(runner, 'rank', 0)
        if rank != 0:
            return
        m = _unwrap(runner.model)
        device = next(m.parameters()).device
        m.load_state_dict(
            {k: v.to(device) for k, v in self.avg_model.module.state_dict().items()})
        out = osp.join(runner.work_dir, 'swa_model.pth')
        save_checkpoint(
            runner.model,
            out,
            optimizer=None,
            meta=dict(
                swa=True,
                swa_start=self.swa_start,
                epoch=runner.epoch + 1,
                config=runner.meta.get('config', '') if runner.meta else ''))
        runner.logger.info(f'[SWAHook] Saved SWA weights to {out}')


@HOOKS.register_module()
class SWARestoreHook(Hook):
    """在 EvalHook（priority≈50）之后执行，恢复 ``SWAHook`` 为评测备份的训练权重。

    解决最后一轮训练后不再进入 ``before_train_epoch`` 导致权重一直为 SWA 的问题。

    ``priority`` 仅在 config 的 hook 字典中设置（如 55），勿在 ``__init__`` 里赋给实例。
    """

    def __init__(self):
        pass

    def after_train_epoch(self, runner):
        backup = getattr(runner, '_swa_train_backup', None)
        if backup is None:
            return
        m = _unwrap(runner.model)
        device = next(m.parameters()).device
        m.load_state_dict({k: v.to(device) for k, v in backup.items()})
        runner._swa_train_backup = None
