import os.path as osp
import random

import mmcv
import numpy as np
from mmcv.utils import print_log

from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class EarthMapOEM8OversampleDataset(CustomDataset):
    """EarthMap OEM8 训练集：以一定概率改为抽取「含 Bareland」的瓦片。

    磁盘标签为 OEM trainId（Bareland=1，0=无效）。与 ``reduce_zero_label=True`` 的
    pipeline 一致，此处按原始 trainId==1 统计像素数。
    """

    def __init__(self,
                 bareland_oversample_prob=0.5,
                 min_bareland_pixels=100,
                 bareland_train_id=1,
                 **kwargs):
        self.bareland_oversample_prob = bareland_oversample_prob
        self.min_bareland_pixels = min_bareland_pixels
        self.bareland_train_id = bareland_train_id
        super().__init__(**kwargs)
        self.bareland_indices = []
        if not self.test_mode and self.ann_dir is not None:
            self.bareland_indices = self._scan_bareland()
            print_log(
                f'EarthMapOEM8OversampleDataset: {len(self.bareland_indices)}/'
                f'{len(self.img_infos)} tiles with >= {self.min_bareland_pixels} '
                f'Bareland pixels (trainId={self.bareland_train_id})',
                logger=get_root_logger())

    def _scan_bareland(self):
        idx_list = []
        for i, info in enumerate(self.img_infos):
            seg_path = osp.join(self.ann_dir, info['ann']['seg_map'])
            try:
                m = mmcv.imread(seg_path, flag='unchanged', backend='pillow')
            except Exception:
                continue
            if m is None:
                continue
            if m.ndim == 3:
                m = m[..., 0]
            cnt = int((m.astype(np.int64) == self.bareland_train_id).sum())
            if cnt >= self.min_bareland_pixels:
                idx_list.append(i)
        return idx_list

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        if (self.bareland_indices and random.random() < self.bareland_oversample_prob):
            idx = random.choice(self.bareland_indices)
        return self.prepare_train_img(idx)
