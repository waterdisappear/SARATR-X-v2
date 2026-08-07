# Copyright (c) OpenMMLab. All rights reserved.
"""Extra EvalHook that logs metrics with a prefix (e.g. val vs test)."""
from mmdet.core import DistEvalHook, EvalHook


class TestEvalHook(EvalHook):
    """EvalHook 子类：将评估结果以指定前缀写入 log（用于同时看 val 和 test）。"""

    def __init__(self, dataloader, metric_key_prefix='test_', save_best=None, **kwargs):
        # mmcv EvalHook 要求 save_best 为 str 或 None，不能为 bool
        super().__init__(dataloader, save_best=save_best, **kwargs)
        self.metric_key_prefix = metric_key_prefix

    def evaluate(self, runner, results):
        """评估并将指标以 prefix 写入 runner.log_buffer.output。"""
        kw = dict(self.eval_kwargs)
        per_cls = kw.pop('per_class_table', True)
        eval_results = self.dataloader.dataset.evaluate(
            results,
            logger=runner.logger,
            per_class_table=per_cls,
            **kw)
        for key, val in eval_results.items():
            runner.log_buffer.output[self.metric_key_prefix + key] = val
        key_indicator = getattr(self, 'key_indicator', None)
        return eval_results.get(key_indicator) if key_indicator else None


class DistTestEvalHook(DistEvalHook):
    """分布式下的 TestEvalHook。"""

    def __init__(self, dataloader, metric_key_prefix='test_', save_best=None, **kwargs):
        # mmcv EvalHook 要求 save_best 为 str 或 None
        super().__init__(dataloader, save_best=save_best, **kwargs)
        self.metric_key_prefix = metric_key_prefix

    def evaluate(self, runner, results):
        # 仅 rank0 打 logger，避免多卡重复打印 per-class 表格
        _logger = runner.logger if runner.rank == 0 else None
        kw = dict(self.eval_kwargs)
        per_cls = kw.pop('per_class_table', True)
        eval_results = self.dataloader.dataset.evaluate(
            results,
            logger=_logger,
            per_class_table=per_cls,
            **kw)
        if runner.rank == 0:
            for key, val in eval_results.items():
                runner.log_buffer.output[self.metric_key_prefix + key] = val
        key_indicator = getattr(self, 'key_indicator', None)
        return eval_results.get(key_indicator) if key_indicator else None
