import mmcv
import numpy as np
import os.path as osp
from PIL import Image

from mmseg.datasets.builder import PIPELINES


@PIPELINES.register_module()
class SETR_Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio range
    and multiply it with the image scale.

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range.

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 crop_size=None,
                 setr_multi_scale=False):

        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            # assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.crop_size = crop_size
        self.setr_multi_scale = setr_multi_scale

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""

        if self.keep_ratio:
            if self.setr_multi_scale:
                if min(results['scale']) < self.crop_size[0]:
                    new_short = self.crop_size[0]
                else:
                    new_short = min(results['scale'])
                    
                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)

            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module()
class LoadHHAs3ChSAR(object):
    """Load HH tiff as 3-channel image with nonlinear SAR quantization.

    Steps:
    1) Read HH single-channel 16-bit tiff.
    2) Clip by percentile to suppress strong outliers.
    3) Apply nonlinear log quantization to [0, 255].
    4) Repeat channel to 3 channels for RGB-style backbones.
    """

    def __init__(self, to_float32=False, clip_percentile=99.5, log_gain=25.0):
        self.to_float32 = to_float32
        self.clip_percentile = clip_percentile
        self.log_gain = log_gain

    def _nonlinear_quantize(self, img):
        img = img.astype(np.float32)
        if self.clip_percentile is not None:
            upper = np.percentile(img, self.clip_percentile)
        else:
            upper = img.max()
        upper = max(float(upper), 1e-6)
        img = np.clip(img, 0, upper) / upper
        img = np.log1p(self.log_gain * img) / np.log1p(self.log_gain)
        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)

    def __call__(self, results):
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = np.array(Image.open(filename))
        if img.ndim == 3:
            img = img[..., 0]

        img = self._nonlinear_quantize(img)
        img = np.stack([img, img, img], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(to_float32={self.to_float32}, '
                f'clip_percentile={self.clip_percentile}, '
                f'log_gain={self.log_gain})')


@PIPELINES.register_module()
class LoadPreprocessedGrayAs3Ch(object):
    """Load already-preprocessed SAR / grayscale as 3-channel, no percentile or log.

    Expects uint8 (or numeric) tiles under ``img_dir``; values are copied to
    three identical channels for RGB-style backbones. Use when data were
    normalized or enhanced offline.
    """

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = np.array(Image.open(filename))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = np.concatenate([img, img, img], axis=-1)
            else:
                img = img[:, :, :3]

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(to_float32={self.to_float32})'


@PIPELINES.register_module()
class LoadPolSARAmplitudeRGB(object):
    """Load HH/HV/VV amplitude images as pseudo RGB input.

    The dataset should provide HH files as main image entries (e.g. in `hh/`).
    This loader automatically finds the paired HV/VV amplitude files using the
    shared patch id and stacks channels as:
        R <- HH_amp, G <- HV_amp, B <- VV_amp
    """

    def __init__(self, to_float32=False, clip_percentile=None, log_gain=15.0):
        self.to_float32 = to_float32
        self.clip_percentile = clip_percentile
        self.log_gain = log_gain

    def _nonlinear_quantize(self, img):
        img = img.astype(np.float32)
        upper = np.percentile(img, self.clip_percentile)
        upper = max(float(upper), 1e-6)
        img = np.clip(img, 0, upper) / upper
        img = np.log1p(self.log_gain * img) / np.log1p(self.log_gain)
        return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)

    @staticmethod
    def _build_pair_path(hh_path, pol):
        hh_name = osp.basename(hh_path)
        hh_dir = osp.dirname(hh_path)

        # AIR-PolSAR-Seg 2.0 format:
        # .../<split>/hh/<patch>_hh_amp.tiff -> .../<split>/{hv,vv}/<patch>_{hv,vv}_amp.tiff
        if hh_name.endswith('_hh_amp.tiff'):
            split_dir = osp.dirname(hh_dir)
            patch_id = hh_name[:-len('_hh_amp.tiff')]
            return osp.join(split_dir, pol, f'{patch_id}_{pol}_amp.tiff')

        # AIR-PolSAR-Seg 1.0 format:
        # .../<split>/AIR-PolarSAR-Seg-xxx_HH.tiff -> .../<split>/AIR-PolarSAR-Seg-xxx_{HV,VV}.tiff
        if hh_name.endswith('_HH.tiff'):
            return osp.join(hh_dir, hh_name.replace('_HH.tiff', f'_{pol.upper()}.tiff'))

        raise ValueError(
            f'Unexpected HH filename: {hh_name}, expected *_hh_amp.tiff or *_HH.tiff')

    def __call__(self, results):
        if results.get('img_prefix') is not None:
            hh_path = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            hh_path = results['img_info']['filename']

        hv_path = self._build_pair_path(hh_path, 'hv')
        vv_path = self._build_pair_path(hh_path, 'vv')

        hh = np.array(Image.open(hh_path))
        hv = np.array(Image.open(hv_path))
        vv = np.array(Image.open(vv_path))

        if hh.ndim == 3:
            hh = hh[..., 0]
        if hv.ndim == 3:
            hv = hv[..., 0]
        if vv.ndim == 3:
            vv = vv[..., 0]

        img = np.stack([hh, hv, vv], axis=-1)
        if self.clip_percentile is not None:
            img = np.stack(
                [self._nonlinear_quantize(img[..., i]) for i in range(3)],
                axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = hh_path
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(to_float32={self.to_float32}, '
                f'clip_percentile={self.clip_percentile}, '
                f'log_gain={self.log_gain})')
