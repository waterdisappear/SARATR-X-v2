# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp

import numpy as np

from .builder import ROTATED_DATASETS
from .dota import DOTADataset
from mmrotate.core import poly2obb_np


@ROTATED_DATASETS.register_module()
class SARDataset(DOTADataset):
    """SAR ship dataset for detection (Support RSSDD and HRSID)."""
    CLASSES = ('ship', )
    PALETTE = [
        (0, 255, 0),
    ]


@ROTATED_DATASETS.register_module()
class RSARDataset(DOTADataset):
    """RSAR dataset for rotated SAR object detection.

    RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark.
    Ref: https://github.com/zhasion/RSAR
    标注格式为 DOTA，类别可为全称或缩写: ship/SH, aircraft/AI, car/CA,
    tank/TA, bridge/BR, harbor/HA.
    """
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
    # 标注中可能使用的缩写
    CLASSES_ALIAS = {
        'SH': 0, 'AI': 1, 'CA': 2, 'TA': 3, 'BR': 4, 'HA': 5,
        'ship': 0, 'aircraft': 1, 'car': 2, 'tank': 3, 'bridge': 4, 'harbor': 5,
    }
    PALETTE = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]

    def load_annotations(self, ann_folder):
        """Load RSAR annotations, support both full names and abbreviations."""
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:
            ann_files = glob.glob(ann_folder + '/*.png')
            if not ann_files:
                ann_files = glob.glob(ann_folder + '/*.jpg')
            for ann_file in ann_files:
                ext = osp.splitext(osp.basename(ann_file))[1]
                data_info = {}
                img_id = osp.split(ann_file)[1][:-len(ext)]
                data_info['filename'] = img_id + ext
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            img_dir = None
            if getattr(self, 'data_root', None) and getattr(self, 'img_prefix', None):
                img_dir = osp.join(self.data_root, self.img_prefix)
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                ext = '.png'
                if img_dir and osp.isdir(img_dir):
                    for e in ('.jpg', '.jpeg', '.bmp', '.png'):
                        if osp.isfile(osp.join(img_dir, img_id + e)):
                            ext = e
                            break
                data_info['filename'] = img_id + ext
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        if len(bbox_info) < 10:
                            continue
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except Exception:
                            continue
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        if cls_name not in self.CLASSES_ALIAS:
                            continue
                        label = self.CLASSES_ALIAS[cls_name]
                        if difficulty <= self.difficulty:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8), dtype=np.float32)
                data_infos.append(data_info)
        self.data_infos = data_infos
        self.img_ids = [osp.splitext(x['filename'])[0] for x in data_infos]
        return data_infos
