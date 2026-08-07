# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile

from ..builder import ROTATED_PIPELINES

# 当默认后缀找不到文件时，尝试的后缀列表（如 RSAR 图像可能为 .bmp）
IMAGE_EXTENSIONS_FALLBACK = ('.bmp', '.png', '.jpg', '.jpeg')


@ROTATED_PIPELINES.register_module()
class LoadImageFromFileMultiExt(LoadImageFromFile):
    """Load image from file; if the path does not exist, try same basename with
    other extensions (e.g. .bmp, .png, .jpg) so that datasets with mixed
    extensions (e.g. RSAR) work without changing annotations."""

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        img_prefix = results.get('img_prefix')
        img_info = results['img_info']
        orig_filename = img_info['filename']
        if img_prefix is not None:
            filename = osp.join(img_prefix, orig_filename)
        else:
            filename = orig_filename

        try:
            img_bytes = self.file_client.get(filename)
        except FileNotFoundError:
            dirname = osp.dirname(filename)
            base = osp.splitext(osp.basename(filename))[0]
            for ext in IMAGE_EXTENSIONS_FALLBACK:
                candidate = osp.join(dirname, base + ext)
                try:
                    img_bytes = self.file_client.get(candidate)
                    filename = candidate
                    img_info['filename'] = base + ext
                    break
                except FileNotFoundError:
                    continue
            else:
                raise FileNotFoundError(
                    f'No image found for {orig_filename} in {dirname} '
                    f'(tried {IMAGE_EXTENSIONS_FALLBACK})')

        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = img_info['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results
