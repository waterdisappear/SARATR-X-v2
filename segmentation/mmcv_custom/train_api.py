import os.path as osp
import random
import warnings

import numpy as np
import mmcv_custom.val_test_eval_hooks as _valtest_hooks
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import build_optimizer, build_runner

from mmcv_custom.val_test_eval_hooks import (
    ValTestDistEvalHook,
    ValTestEvalHook,
)
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
try:
    import apex
except Exception:
    apex = None


def _ensure_safe_text_logger(runner, cfg, logger):
    """去掉 mmcv 文本日志器，只保留 SafeTextLoggerHook。

    仅用 ``type(h) is TextLoggerHook`` 不够：部分环境会注册 TextLoggerHook 的其它子类，
    仍会按 log_dict 插入顺序打印，出现 ``mIoU, mAcc, OA`` 而非论文顺序。
    """
    from mmcv.runner.hooks.logger import TextLoggerHook as MMCVTextLoggerHook

    from mmcv_custom.safe_text_logger import SafeTextLoggerHook

    def _drop_non_safe_text(h):
        return isinstance(h, MMCVTextLoggerHook) and not isinstance(
            h, SafeTextLoggerHook)

    before = len(runner._hooks)
    runner._hooks = [h for h in runner._hooks if not _drop_non_safe_text(h)]
    removed = before - len(runner._hooks)
    if removed:
        logger.info(
            'train_api: 已移除 %d 个非 SafeTextLoggerHook 的文本日志 Hook',
            removed)

    has_safe = any(isinstance(h, SafeTextLoggerHook) for h in runner._hooks)
    if not has_safe:
        lc = cfg.get('log_config') or {}
        interval = lc.get('interval', 50)
        by_epoch = False
        for hi in lc.get('hooks', []):
            if isinstance(hi, dict) and 'by_epoch' in hi:
                by_epoch = bool(hi['by_epoch'])
                break
        runner.register_hook(
            SafeTextLoggerHook(by_epoch=by_epoch, interval=interval),
            priority='VERY_LOW')
        logger.info(
            'train_api: 已补注册 SafeTextLoggerHook(interval=%s, by_epoch=%s)',
            interval, by_epoch)


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)
    logger.info(
        'mmcv_custom.train_segmentor (%s); eval hooks from val_test_eval_hooks if validate=True.',
        osp.abspath(__file__))

    _raw_eval = cfg.get('evaluation', {}) or {}
    if not validate and _raw_eval.get('eval_test'):
        logger.warning(
            'train_api: 配置里 evaluation.eval_test=True，但当前 validate=False（'
            '例如使用了 --no-validate），不会注册 ValTestEvalHook，也不会有 test 指标。')

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # use apex fp16 optimizer
    if cfg.optimizer_config.get("type", None) and cfg.optimizer_config["type"] == "DistOptimizerHook":
        if cfg.optimizer_config.get("use_fp16", False):
            if apex is None:
                logger.warning('apex is not installed, fallback to FP32 training.')
                cfg.optimizer_config["use_fp16"] = False
            else:
                model, optimizer = apex.amp.initialize(
                    model.cuda(), optimizer, opt_level="O1")
                for m in model.modules():
                    if hasattr(m, "fp16_enabled"):
                        m.fp16_enabled = True

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    _ensure_safe_text_logger(runner, cfg, logger)

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        _dr = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            _dr = torch.distributed.get_rank()
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = dict(cfg.get('evaluation', {}) or {})
        eval_cfg.pop('_delete_', None)
        eval_test = bool(eval_cfg.pop('eval_test', False)) or bool(
            cfg.get('eval_test', False))
        # ValTestHook 内已对 single/multi_gpu_test 强制 pre_eval=False；
        # 此处写入 kwargs，避免与 mmseg 默认 / base 里 True 合并后误导排查。
        eval_cfg['pre_eval'] = False
        # 与官方 mmseg 一致：仅 EpochBasedRunner 按 epoch 评估；IterBasedRunnerAmp 等按 iter。
        _rt = str(cfg.runner.get('type', ''))
        eval_cfg['by_epoch'] = 'EpochBasedRunner' in _rt
        if _dr == 0:
            logger.info(
                'ValTest: evaluation by_epoch=%s runner.type=%s',
                eval_cfg['by_epoch'], cfg.runner.get('type'))
        test_dataloader = None
        test_dataloader_single_gpu = None
        if eval_test:
            test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
            test_dataloader = build_dataloader(
                test_dataset,
                samples_per_gpu=1,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)
            if distributed:
                test_dataloader_single_gpu = build_dataloader(
                    test_dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    dist=False,
                    shuffle=False)
                logger.info(
                    'ValTest: single-GPU test DataLoader batches=%d (rank0 eval).',
                    len(test_dataloader_single_gpu))
            logger.info(
                'ValTest eval: test set size=%d (eval_test=True).',
                len(test_dataset))
        eval_hook = ValTestDistEvalHook if distributed else ValTestEvalHook
        if _dr == 0:
            logger.info(
                'ValTest: hook_impl class=%s module=%s file=%s eval_test=%s',
                eval_hook.__name__, eval_hook.__module__,
                _valtest_hooks.__file__, eval_test)
            if eval_test and test_dataloader is None:
                logger.warning(
                    'ValTest: eval_test=True but test_dataloader is None '
                    '(check cfg.evaluation merge / eval_test key).')
            elif eval_test and test_dataloader is not None:
                logger.info(
                    'ValTest: test_dataloader batches=%d.',
                    len(test_dataloader))
        if distributed:
            _eval_hook_inst = eval_hook(
                val_dataloader,
                test_dataloader=test_dataloader,
                test_dataloader_single_gpu=test_dataloader_single_gpu,
                eval_test=eval_test,
                **eval_cfg)
        else:
            _eval_hook_inst = eval_hook(
                val_dataloader,
                test_dataloader=test_dataloader,
                test_dataloader_single_gpu=test_dataloader,
                eval_test=eval_test,
                **eval_cfg)
        runner.register_hook(_eval_hook_inst, priority='LOW')
        if _dr == 0:
            _wd_abs = osp.abspath(cfg.work_dir)
            logger.info(
                'ValTest: valtest_trace.log / eval_test_history.jsonl / '
                'valtest_hook_registered.txt 目录(与训练 log 同目录): %s',
                _wd_abs)
            try:
                _flag = osp.join(_wd_abs, 'valtest_hook_registered.txt')
                with open(_flag, 'w', encoding='utf-8') as f:
                    f.write('work_dir=%s\n' % _wd_abs)
                    f.write('hook=%s\n' % type(_eval_hook_inst).__name__)
                    f.write('eval_test=%s\n' % eval_test)
                    f.write('val_test_eval_hooks=%s\n' % _valtest_hooks.__file__)
            except OSError as exc:
                logger.warning('ValTest: 无法写入 %s: %s', _flag, exc)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
