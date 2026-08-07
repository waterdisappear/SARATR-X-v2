# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import mmdet

from .core import *  # noqa: F401, F403
from .datasets import *  # noqa: F401, F403
from .models import *  # noqa: F401, F403
from .version import __version__, short_version


def digit_version(version_str):
    """Digit version."""
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


# 兼容 MMCV 1.x (1.5.3~1.8.0) 与 2.x (2.0.0rc4~2.1.0)，仅警告不拦截
mmcv_version = digit_version(mmcv.__version__)
_mmcv_1x_ok = (mmcv_version >= digit_version('1.5.3')
               and mmcv_version <= digit_version('1.8.0'))
_mmcv_2x_ok = (mmcv_version >= digit_version('2.0.0rc4')
               and mmcv_version <= digit_version('2.1.0'))
if not (_mmcv_1x_ok or _mmcv_2x_ok):
    warnings.warn(
        f'MMCV=={mmcv.__version__} is not in the recommended range '
        '[1.5.3, 1.8.0] or [2.0.0rc4, 2.1.0]. You may meet compatibility issues.',
        UserWarning,
        stacklevel=2)

mmdet_minimum_version = '2.25.1'
mmdet_maximum_version = '3.0.0'
mmdet_version = digit_version(mmdet.__version__)

assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version < digit_version(mmdet_maximum_version)), \
    f'MMDetection=={mmdet.__version__} is used but incompatible. ' \
    f'Please install mmdet>={mmdet_minimum_version}, <{mmdet_maximum_version}.'

__all__ = ['__version__', 'short_version']
