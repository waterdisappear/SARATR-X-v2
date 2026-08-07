from mmcv.runner import HOOKS
from mmcv.runner.hooks.logger import TextLoggerHook


@HOOKS.register_module()
class SafeTextLoggerHook(TextLoggerHook):
    """Text logger hook tolerant to missing time fields.

    Some custom training flows may not populate ``time``/``data_time`` for
    every iteration. The upstream TextLoggerHook assumes these keys always
    exist and may raise KeyError. This hook fills defaults before logging.

    For validation lines (``Iter(val)``), metrics are printed in paper-friendly
    order: OA, Kappa, mIoU, mAcc; if only mmseg defaults exist, ``aAcc`` is
    labeled as OA.
    """

    _VAL_SKIP_KEYS = frozenset({
        'mode',
        'Epoch',
        'iter',
        'lr',
        'time',
        'data_time',
        'memory',
        'epoch',
    })

    def _log_info(self, log_dict, runner):
        if log_dict.get('mode') == 'train':
            log_dict.setdefault('time', 0.0)
            log_dict.setdefault('data_time', 0.0)
            super()._log_info(log_dict, runner)
            return

        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                runner.logger.info('Exp name: %s', runner.meta['exp_name'])

        if self.by_epoch:
            log_str = (f'Epoch({log_dict["mode"]}) '
                       f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t')
        else:
            log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = []
        seen = set()

        def _append(display_name, val):
            try:
                num = float(val)
            except (TypeError, ValueError):
                log_items.append(f'{display_name}: {val}')
            else:
                log_items.append(f'{display_name}: {num:.4f}')

        def _paper_tuple(key_prefix):
            """OA 优先用显式键；mmseg 常只有 aAcc 无 OA，需与 mIoU/mAcc 一起按论文顺序输出。"""
            oa = log_dict.get(key_prefix + 'OA')
            if oa is None:
                oa = log_dict.get(key_prefix + 'aAcc')
            return (
                oa,
                log_dict.get(key_prefix + 'Kappa'),
                log_dict.get(key_prefix + 'mIoU'),
                log_dict.get(key_prefix + 'mAcc'),
            )

        oa, kappa, miou, macc = _paper_tuple('')
        for disp, val in (
                ('OA', oa),
                ('Kappa', kappa),
                ('mIoU', miou),
                ('mAcc', macc),
        ):
            if val is not None:
                _append(disp, val)
        for k in ('OA', 'Kappa', 'mIoU', 'mAcc', 'aAcc'):
            if k in log_dict:
                seen.add(k)

        if any(
                log_dict.get('test_' + x) is not None
                for x in ('OA', 'aAcc', 'Kappa', 'mIoU', 'mAcc')):
            toa, tkappa, tmiou, tmacc = _paper_tuple('test_')
            for disp, val in (
                    ('test_OA', toa),
                    ('test_Kappa', tkappa),
                    ('test_mIoU', tmiou),
                    ('test_mAcc', tmacc),
            ):
                if val is not None:
                    _append(disp, val)
            for k in (
                    'test_OA', 'test_Kappa', 'test_mIoU', 'test_mAcc',
                    'test_aAcc',
            ):
                if k in log_dict:
                    seen.add(k)

        for name, val in log_dict.items():
            if name in self._VAL_SKIP_KEYS or name in seen:
                continue
            _append(name, val)
            seen.add(name)

        log_str += ', '.join(log_items)
        runner.logger.info(log_str)
