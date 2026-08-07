# Copyright (c) OpenMMLab. All rights reserved.
"""MMRotate 版类别数检查 Hook。

mmdet ``NumClassCheckHook`` 在遍历 ``named_modules()`` 时，仅排除 ``RPNHead``；
``RotatedRPNHead`` 继承 ``AnchorHead`` 而非 ``RPNHead``，其 ``num_classes=1``
（RPN 前后景）会与 ``len(dataset.CLASSES)`` 不一致而误触发断言。

本 Hook 在相同逻辑上额外跳过 ``RotatedRPNHead``。
"""
from mmcv.cnn import VGG
from mmcv.runner import HOOKS, Hook
from mmdet.models.dense_heads import GARPNHead, RPNHead
from mmdet.models.roi_heads.mask_heads import FusedSemanticHead


@HOOKS.register_module()
class MMRotateNumClassCheckHook(Hook):
    """与 mmdet ``NumClassCheckHook`` 一致，但跳过 ``RotatedRPNHead``。"""

    def _check_head(self, runner):
        model = runner.model
        dataset = runner.data_loader.dataset
        if dataset.CLASSES is None:
            runner.logger.warning(
                f'Please set `CLASSES` in the {dataset.__class__.__name__} and '
                f'check if it is consistent with the `num_classes` of head')
        else:
            assert type(dataset.CLASSES) is not str, (
                f'`CLASSES` in {dataset.__class__.__name__} should be a tuple of '
                f'str. Add comma if number of classes is 1 as '
                f'CLASSES = ({dataset.CLASSES},)')
            for name, module in model.named_modules():
                if not hasattr(module, 'num_classes'):
                    continue
                if isinstance(
                        module,
                    (RPNHead, VGG, FusedSemanticHead, GARPNHead)):
                    continue
                # Rotated RPN：仍为二分类头，num_classes 固定为 1
                if module.__class__.__name__ == 'RotatedRPNHead':
                    continue
                assert module.num_classes == len(dataset.CLASSES), (
                    f'The `num_classes` ({module.num_classes}) in '
                    f'{module.__class__.__name__} of {model.__class__.__name__} '
                    f'does not matches the length of `CLASSES` '
                    f'({len(dataset.CLASSES)}) in {dataset.__class__.__name__}')

    def before_train_epoch(self, runner):
        self._check_head(runner)

    def before_val_epoch(self, runner):
        self._check_head(runner)
