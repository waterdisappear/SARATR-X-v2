# -*- coding: utf-8 -*-
"""Validation + optional test-set eval with mIoU, OA (aAcc), and multiclass Kappa."""

import inspect
import os.path as osp
from collections import OrderedDict

import torch.distributed as dist
from mmcv.runner import get_dist_info
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv_custom.rs_metrics import (
    accumulate_confusion_matrix,
    is_pre_eval_format,
    kappa_from_confusion,
    overall_accuracy_from_confusion,
)

try:
    from mmseg.core.evaluation.eval_hooks import DistEvalHook as _MMSegDistEvalHook
    from mmseg.core.evaluation.eval_hooks import EvalHook as _MMSegEvalHook
except Exception:  # pragma: no cover - import path varies by mmseg version
    from mmseg.core.evaluation import DistEvalHook as _MMSegDistEvalHook
    from mmseg.core.evaluation import EvalHook as _MMSegEvalHook

try:
    from mmcv.runner.hooks.logger import LoggerHook
except Exception:  # pragma: no cover
    LoggerHook = None


def _single_gpu_test_compat(model, data_loader, show=False):
    """mmseg 0.11 原版无 ``pre_eval`` 参数；新版本有，此处统一在支持时传 ``False``。"""
    from mmseg.apis import single_gpu_test

    kw = {}
    if 'pre_eval' in inspect.signature(single_gpu_test).parameters:
        kw['pre_eval'] = False
    return single_gpu_test(model, data_loader, show=show, **kw)


def _multi_gpu_test_compat(model, data_loader, tmpdir, gpu_collect):
    from mmseg.apis import multi_gpu_test

    kw = {}
    if 'pre_eval' in inspect.signature(multi_gpu_test).parameters:
        kw['pre_eval'] = False
    return multi_gpu_test(
        model, data_loader, tmpdir=tmpdir, gpu_collect=gpu_collect, **kw)


def _as_log_scalar(val):
    """写入 log_buffer 时用原生 float，避免 mmcv 1.3 / numpy 标量在合并 log_dict 时丢键。"""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return val


def _paper_scalar_metrics(eval_res):
    """Paper-style summary scalars (order: OA, Kappa, mIoU, mAcc).

    OA: overall accuracy (与 mmseg aAcc 一致定义，优先用混淆矩阵 OA)。
    Kappa: 多类 Kappa（与 Kappa 系数论文定义一致）。
    mIoU: 各类 IoU 均值。
    mAcc: 各类像素精度均值（mmseg mAcc）。
    """
    oa = eval_res.get('OA')
    if oa is None:
        oa = eval_res.get('aAcc')
    kappa = eval_res.get('Kappa')
    miou = eval_res.get('mIoU')
    macc = eval_res.get('mAcc')
    od = OrderedDict()
    if oa is not None:
        od['OA'] = oa
    if kappa is not None:
        od['Kappa'] = kappa
    if miou is not None:
        od['mIoU'] = miou
    if macc is not None:
        od['mAcc'] = macc
    return od


def _save_best_ckpt_simple(runner, key_indicator, key_score, by_epoch):
    """当 mmseg EvalHook 无 ``_save_ckpt`` 时，按 val 指标保存 best 权重（greater 规则）。"""
    import math

    if runner.meta is None:
        return
    msgs = runner.meta.setdefault('hook_msgs', {})
    best = msgs.get('best_score', -math.inf)
    try:
        score = float(key_score)
    except (TypeError, ValueError):
        return
    if score <= best:
        return
    msgs['best_score'] = score
    cur = f'iter_{runner.iter + 1}' if not by_epoch else f'epoch_{runner.epoch + 1}'
    name = f'best_{key_indicator}_{cur}.pth'
    runner.save_checkpoint(runner.work_dir, name, create_symlink=False)
    msgs['best_ckpt'] = osp.join(runner.work_dir, name)
    runner.logger.info(
        'ValTest: saved new best checkpoint %s (score=%.6f)', name, score)


def _log_paper_metrics_line(runner, split_tag, eval_res):
    """单行论文指标（与 Iter(val) 摘要一致顺序）。"""
    oa = eval_res.get('OA')
    if oa is None:
        oa = eval_res.get('aAcc')
    kappa = eval_res.get('Kappa')
    miou = eval_res.get('mIoU', 0.0)
    macc = eval_res.get('mAcc')
    if kappa is None:
        runner.logger.info(
            '[%s] 论文指标  OA=%.4f  Kappa=N/A  mIoU=%.4f  mAcc=%s',
            split_tag,
            float(oa) if oa is not None else 0.0,
            float(miou),
            ('%.4f' % float(macc)) if macc is not None else 'N/A',
        )
    else:
        runner.logger.info(
            '[%s] 论文指标  OA=%.4f  Kappa=%.4f  mIoU=%.4f  mAcc=%s',
            split_tag,
            float(oa) if oa is not None else 0.0,
            float(kappa),
            float(miou),
            ('%.4f' % float(macc)) if macc is not None else 'N/A',
        )


def _flush_eval_log_buffer(runner):
    """Write log_buffer (val/test metrics) in the same iter as EvalHook.

    mmcv EvalHook clears the training log then fills eval metrics; without an
    extra LoggerHook pass, only part of the buffer may appear in .log / .json.

    注意：不要调用 ``LoggerHook.after_train_iter``：其中会先执行
    ``LogBuffer.average(interval)``（当 ``(iter+1) % log_interval == 0`` 时），
    与手工写入的 ``output`` 混用可能打乱指标；应对 **TextLoggerHook** 直接 ``log()``，
    再按 ``reset_flag`` 清 ``output``（与 mmcv 一致）。
    """
    from mmcv.runner.hooks.logger import TextLoggerHook as MMTextLoggerHook

    if LoggerHook is None:
        return
    rank, _ = get_dist_info()
    if rank != 0 or not getattr(runner.log_buffer, 'ready', False):
        return
    for hook in runner._hooks:
        if isinstance(hook, MMTextLoggerHook):
            hook.log(runner)
            if getattr(hook, 'reset_flag', False):
                runner.log_buffer.clear_output()


class ValTestEvalHook(_MMSegEvalHook):
    """Single-GPU: eval val; if ``test_dataloader`` set, also eval test (same iter)."""

    greater_keys = ['mIoU', 'mAcc', 'aAcc', 'OA', 'Kappa']

    def __init__(
            self,
            dataloader,
            test_dataloader=None,
            test_dataloader_single_gpu=None,
            eval_test=False,
            **kwargs):
        self.test_dataloader = test_dataloader
        self.test_dataloader_single_gpu = test_dataloader_single_gpu
        self.eval_test = bool(eval_test)
        super(ValTestEvalHook, self).__init__(dataloader, **kwargs)

    def _evaluate_split(self, runner, results, dataset, split_tag):
        if is_pre_eval_format(results):
            runner.logger.warning(
                'ValTest: pre_eval=True results skip Kappa/OA(confusion); '
                'use pre_eval=False for full metrics.')
            eval_res = dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs)
        else:
            eval_res = dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs)
            cm = accumulate_confusion_matrix(
                results,
                dataset,
                num_classes=len(dataset.CLASSES),
                ignore_index=getattr(dataset, 'ignore_index', 255),
                reduce_zero_label=getattr(dataset, 'reduce_zero_label', False),
                label_map=getattr(dataset, 'label_map', None) or {},
            )
            kappa = kappa_from_confusion(cm)
            oa_cm = overall_accuracy_from_confusion(cm)
            eval_res['Kappa'] = kappa
            eval_res['OA'] = oa_cm

        _log_paper_metrics_line(runner, split_tag, eval_res)

        prefix = '' if split_tag == 'val' else f'{split_tag}_'
        for name, val in _paper_scalar_metrics(eval_res).items():
            runner.log_buffer.output[prefix + name] = _as_log_scalar(val)
        runner.log_buffer.ready = True

        if split_tag == 'val':
            return self._key_score_for_save_best(eval_res)
        return None

    def _save_best_key(self):
        key = getattr(self, 'save_best', None)
        if key is None and isinstance(getattr(self, 'eval_kwargs', None), dict):
            key = self.eval_kwargs.get('save_best')
        return key

    def _key_score_for_save_best(self, eval_res):
        key = self._save_best_key()
        if not key:
            return None
        if key == 'auto':
            return next(iter(eval_res.values()))
        return eval_res.get(key)

    def evaluate(self, runner, results):
        """mmseg ``EvalHook.after_train_iter`` 在 ``single_gpu_test(val)`` 之后调用本方法。

        官方实现只跑 val；此处追加 test，并写日志。注意：mmseg 父类 **从不** 调用
        ``_do_evaluate``，因此 test 逻辑必须放在 ``evaluate`` 中。
        """
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

        key_score = self._evaluate_split(
            runner, results, self.dataloader.dataset, 'val')

        if self.eval_test and self.test_dataloader is not None:
            tl = self.test_dataloader_single_gpu or self.test_dataloader
            results_t = _single_gpu_test_compat(
                runner.model, tl, show=False)
            self._evaluate_split(
                runner, results_t, tl.dataset, 'test')
        elif self.eval_test and self.test_dataloader is None:
            runner.logger.warning(
                'ValTest: eval_test=True but test_dataloader is None; skip test.')

        save_key = self._save_best_key()
        if save_key and key_score is not None:
            if hasattr(self, '_save_ckpt'):
                self._save_ckpt(runner, key_score)
            else:
                _save_best_ckpt_simple(runner, save_key, key_score, self.by_epoch)

        _flush_eval_log_buffer(runner)


class ValTestDistEvalHook(_MMSegDistEvalHook):
    """Distributed val (all ranks); test eval on rank0 single-GPU loader only."""

    greater_keys = ['mIoU', 'mAcc', 'aAcc', 'OA', 'Kappa']

    def __init__(
            self,
            dataloader,
            test_dataloader=None,
            test_dataloader_single_gpu=None,
            eval_test=False,
            **kwargs):
        self.test_dataloader = test_dataloader
        self.test_dataloader_single_gpu = test_dataloader_single_gpu
        self.eval_test = bool(eval_test)
        super(ValTestDistEvalHook, self).__init__(dataloader, **kwargs)

    def _evaluate_split(self, runner, results, dataset, split_tag):
        if is_pre_eval_format(results):
            runner.logger.warning(
                'ValTest: pre_eval=True results skip Kappa/OA(confusion).')
            eval_res = dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs)
        else:
            eval_res = dataset.evaluate(
                results, logger=runner.logger, **self.eval_kwargs)
            cm = accumulate_confusion_matrix(
                results,
                dataset,
                num_classes=len(dataset.CLASSES),
                ignore_index=getattr(dataset, 'ignore_index', 255),
                reduce_zero_label=getattr(dataset, 'reduce_zero_label', False),
                label_map=getattr(dataset, 'label_map', None) or {},
            )
            kappa = kappa_from_confusion(cm)
            oa_cm = overall_accuracy_from_confusion(cm)
            eval_res['Kappa'] = kappa
            eval_res['OA'] = oa_cm

        _log_paper_metrics_line(runner, split_tag, eval_res)

        prefix = '' if split_tag == 'val' else f'{split_tag}_'
        for name, val in _paper_scalar_metrics(eval_res).items():
            runner.log_buffer.output[prefix + name] = _as_log_scalar(val)
        runner.log_buffer.ready = True

        if split_tag == 'val':
            return self._key_score_for_save_best(eval_res)
        return None

    def _save_best_key(self):
        key = getattr(self, 'save_best', None)
        if key is None and isinstance(getattr(self, 'eval_kwargs', None), dict):
            key = self.eval_kwargs.get('save_best')
        return key

    def _key_score_for_save_best(self, eval_res):
        key = self._save_best_key()
        if not key:
            return None
        if key == 'auto':
            return next(iter(eval_res.values()))
        return eval_res.get(key)

    def after_train_iter(self, runner):
        """在父类 ``multi_gpu_test`` 之前可选广播 BN buffer（与旧 ``_do_evaluate`` 意图一致）。"""
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        if getattr(self, 'broadcast_bn_buffer', False):
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)
        super(ValTestDistEvalHook, self).after_train_iter(runner)
        # 所有 rank 必须进入 barrier；不可放在仅 rank0 执行的 ``evaluate`` 内。
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def evaluate(self, runner, results):
        """仅在 rank0 调用（mmseg ``DistEvalHook`` 约定）；在 val 指标后再跑 test。"""
        self.latest_results = results
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

        key_score = self._evaluate_split(
            runner, results, self.dataloader.dataset, 'val')

        if self.eval_test and self.test_dataloader is not None:
            tl = self.test_dataloader_single_gpu or self.test_dataloader
            results_t = _single_gpu_test_compat(
                runner.model, tl, show=False)
            self._evaluate_split(
                runner, results_t, tl.dataset, 'test')
        elif self.eval_test and self.test_dataloader is None:
            runner.logger.warning(
                'ValTest: eval_test=True but test_dataloader is None; skip test.')

        save_key = self._save_best_key()
        if save_key and key_score is not None:
            if hasattr(self, '_save_ckpt'):
                self._save_ckpt(runner, key_score)
            else:
                _save_best_ckpt_simple(runner, save_key, key_score, self.by_epoch)

        _flush_eval_log_buffer(runner)
