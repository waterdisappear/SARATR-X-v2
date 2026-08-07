# Copyright (c) OpenMMLab. All rights reserved.

import os.path as osp
import random
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

from .builder import ROTATED_DATASETS
from .dota import DOTADataset, collect_dota_class_names_from_ann_folder


@ROTATED_DATASETS.register_module()
class FAIRCSARDataset(DOTADataset):
    """A FAIR-CSAR dataset wrapper for rotated SAR detection.

    This dataset dynamically builds ``CLASSES`` by scanning all txt files
    under the given ``ann_file`` folder (DOTA-style polygons).

    Notes:
    - Category names are extracted from token 9 (0-based index 8).
    - Prefer ``label_union_dirs=[train_ann, test_ann]`` so ``CLASSES`` is
      rebuilt at runtime from disk (avoids mmcv config merge overwriting
      ``classes`` with a shorter list). If set, it overrides ``classes``.
    """

    def __init__(self,
                 ann_file: str,
                 pipeline,
                 version: str = 'le90',
                 difficulty: int = 100,
                 classes: Optional[Iterable[str]] = None,
                 label_union_dirs: Optional[Union[str, Sequence[str]]] = None,
                 split: str = 'train',
                 val_ratio: float = 0.0,
                 seed: int = 0,
                 **kwargs):
        if label_union_dirs is not None:
            if isinstance(label_union_dirs, str):
                dirs: List[str] = [label_union_dirs]
            else:
                dirs = list(label_union_dirs)
            if not dirs:
                raise ValueError('label_union_dirs must be non-empty when set.')
            union = set()
            for d in dirs:
                union.update(
                    collect_dota_class_names_from_ann_folder(d, version))
            if not union:
                raise ValueError(
                    f'No class names found under label_union_dirs={dirs!r}.')
            self.CLASSES = tuple(sorted(union))
        elif classes is None:
            scanned = list(
                collect_dota_class_names_from_ann_folder(ann_file, version))
            if not scanned:
                raise ValueError(
                    f'No class names found under ann_folder={ann_file}. '
                    'Make sure it contains DOTA-style *.txt annotations.')
            self.CLASSES = tuple(scanned)
        else:
            self.CLASSES = tuple(classes)

        _want_classes = tuple(self.CLASSES)
        # CustomDataset.get_classes(None) 会用 **类** 上的 ``DOTADataset.CLASSES``（15 类），
        # 必须把最终类别表传给父类。``classes=..., **kwargs`` 时若 kwargs 里还带 classes，
        # 后出现的会覆盖前者，因此把 classes 放进 _kw 并放在最后写入。
        _kw = dict(kwargs)
        _kw.pop('classes', None)
        _kw['classes'] = _want_classes
        super().__init__(ann_file=ann_file,
                         pipeline=pipeline,
                         version=version,
                         difficulty=difficulty,
                         **_kw)

        # 若父类或其它逻辑仍把 CLASSES 改短（与磁盘 FAIR-CSAR 不一致），则强制纠正并重载标注。
        if tuple(self.CLASSES) != _want_classes:
            self._faircsar_repair_classes_and_reload(_want_classes)

        # Random split from one ann_folder into train/val.
        # This is useful when you only have trainval split and want
        # to carve out a small val subset (e.g. 10%).
        if (val_ratio is not None and val_ratio > 0
                and split in ['train', 'val'] and len(self.data_infos) > 0):
            items = []
            for info in self.data_infos:
                img_id = osp.splitext(info['filename'])[0]
                items.append((img_id, info))

            # deterministic base order, then shuffle with seed
            items.sort(key=lambda x: x[0])
            rng = random.Random(seed)
            rng.shuffle(items)

            n_val = int(len(items) * val_ratio)
            if n_val > 0:
                if split == 'val':
                    selected = items[:n_val]
                else:
                    selected = items[n_val:]

                self.data_infos = [info for _, info in selected]
                self.img_ids = [
                    osp.splitext(x['filename'])[0] for x in self.data_infos
                ]
                self.flag = np.zeros(len(self.data_infos),
                                      dtype=np.uint8)

    def _faircsar_repair_classes_and_reload(self, want_classes):
        """Restore CLASSES and rebuild ``data_infos`` (mirrors CustomDataset path)."""
        self.CLASSES = tuple(want_classes)
        ann = self.ann_file
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(ann) as local_path:
                self.data_infos = self.load_annotations(local_path)
        else:
            self.data_infos = self.load_annotations(ann)
        if not self.test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
        self._set_group_flag()

