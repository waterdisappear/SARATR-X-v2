# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import warnings
import zipfile
from collections import OrderedDict, defaultdict
from functools import partial
import unicodedata

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset

from mmcv.utils import print_log

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from mmrotate.core.evaluation.eval_map import print_map_summary
from .builder import ROTATED_DATASETS


def _scalar_ap(ap):
    """AP may be float or 0-dim ndarray from average_precision."""
    if isinstance(ap, np.ndarray):
        return float(ap.ravel()[0])
    return float(ap)


def _ap50_metric_key(cls_name, idx):
    """Log-friendly key: AP50_012_ClassName (non-alnum -> _)."""
    safe = re.sub(r'[^0-9a-zA-Z_]+', '_', str(cls_name)).strip('_')
    if not safe:
        safe = f'cls{idx}'
    return f'AP50_{idx:03d}_{safe}'


def normalize_dota_class_token(raw):
    """Normalize class name token the same way as ``DOTADataset.load_annotations``."""
    cls_name = raw
    cls_name = cls_name.strip().replace('\ufeff', '')
    cls_name = cls_name.replace('\u00A0', ' ')  # NBSP
    cls_name = ''.join(
        ch for ch in cls_name if unicodedata.category(ch) != 'Cf')
    return cls_name.strip()


def collect_dota_class_names_from_ann_folder(ann_folder, version='le90'):
    """Collect unique class names under ``ann_folder`` using the same rules as loading.

    Line parsing matches ``DOTADataset.load_annotations`` (no extra len>=10 gate).
    """
    cls_set = set()
    ann_files = glob.glob(ann_folder + '/*.txt')
    for ann_file in ann_files:
        if os.path.getsize(ann_file) == 0:
            continue
        with open(ann_file) as f:
            for si in f.readlines():
                bbox_info = si.split()
                poly = np.array(bbox_info[:8], dtype=np.float32)
                try:
                    poly2obb_np(poly, version)
                except Exception:
                    continue
                try:
                    cls_name = normalize_dota_class_token(bbox_info[8])
                    int(bbox_info[9])
                except (IndexError, ValueError):
                    continue
                if cls_name:
                    cls_set.add(cls_name)
    return sorted(cls_set)


@ROTATED_DATASETS.register_module()
class DOTADataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 classes=None,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(DOTADataset, self).__init__(
            ann_file, pipeline, classes=classes, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i for i, c in enumerate(self.CLASSES)}
        unknown_cls_names = set()
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        # Class name token might contain BOM / zero-width chars
                        # depending on how txt files were exported.
                        cls_name = normalize_dota_class_token(bbox_info[8])
                        difficulty = int(bbox_info[9])
                        label = cls_map.get(cls_name, None)
                        if label is None:
                            unknown_cls_names.add(cls_name)
                            continue
                        if difficulty > self.difficulty:
                            pass
                        else:
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
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        if unknown_cls_names and len(self.CLASSES) > 0:
            warnings.warn(
                f'Found {len(unknown_cls_names)} annotation class(es) not in '
                f'config CLASSES. Examples: {sorted(list(unknown_cls_names))[:10]}',
                UserWarning)
        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None,
                 nproc=4,
                 per_class_table=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold(s). If list (e.g.
                [0.5, 0.55, ..., 0.95]), mAP = average over these IoUs (like RSAR).
                If float, mAP at that single IoU. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
            per_class_table (bool): 若为 True，在 logger 上打印每类 AP 表格（与整体 mAP），
                并在返回的 dict 中加入每类 ``AP50_xxx`` 标量（仅 ``metric=='mAP50'`` 时）。
            **kwargs: 忽略未知参数（兼容 evaluation 配置透传）。
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'mAP50']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        # 有 logger 且需要表格时让 eval_rbbox_map 内部 print_map_summary；否则静默后再按需补打
        rb_logger = logger if (
            per_class_table and logger not in (None, 'silent')) else 'silent'

        if metric == 'mAP50':
            if per_class_table and logger not in (None, 'silent'):
                print_log(
                    '\n[Per-class mAP50 @ IoU=0.5] 表格末行 mAP 为整体 mAP50',
                    logger=logger)
            mean_ap, cls_details = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=0.5,
                dataset=self.CLASSES,
                logger=rb_logger,
                nproc=nproc)
            eval_results['mAP50'] = round(mean_ap, 3)
            if per_class_table:
                for i, cls_name in enumerate(self.CLASSES):
                    eval_results[_ap50_metric_key(cls_name, i)] = round(
                        _scalar_ap(cls_details[i]['ap']), 4)
                if rb_logger == 'silent' and logger not in (None, 'silent'):
                    print_map_summary(
                        mean_ap,
                        cls_details,
                        self.CLASSES,
                        scale_ranges,
                        logger=logger)
        elif metric == 'mAP':
            iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
            mean_aps = []
            for thr in iou_thrs:
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=thr,
                    dataset=self.CLASSES,
                    logger='silent',
                    nproc=nproc)
                mean_aps.append(mean_ap)
                # 同时返回常用单点阈值（便于输出 mAP50/mAP75 等）
                try:
                    thr_key = int(round(float(thr) * 100))
                    eval_results[f'mAP{thr_key}'] = round(mean_ap, 3)
                except Exception:
                    pass
            eval_results['mAP'] = round(sum(mean_aps) / len(mean_aps), 3)
        else:
            raise NotImplementedError

        return eval_results

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.

        Returns:
            list: merged results.
        """

        def extract_xy(img_id):
            """Extract x and y coordinates from image ID.

            Args:
                img_id (str): ID of the image.

            Returns:
                Tuple of two integers, the x and y coordinates.
            """
            pattern = re.compile(r'__(\d+)___(\d+)')
            match = pattern.search(img_id)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return x, y
            else:
                warnings.warn(
                    "Can't find coordinates in filename, "
                    'the coordinates will be set to (0,0) by default.',
                    category=Warning)
                return 0, 0

        collector = defaultdict(list)
        for idx, img_id in enumerate(self.img_ids):
            result = results[idx]
            oriname = img_id.split('__', maxsplit=1)[0]
            x, y = extract_xy(img_id)
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))
            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Executing on Single Processor')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print(f'Executing on {nproc} processors')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        # Return a zipped list of merged results
        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, 'Task1_' + cls + '.txt')
            for cls in self.CLASSES
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        with zipfile.ZipFile(
                osp.join(out_folder, target_name + '.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results
