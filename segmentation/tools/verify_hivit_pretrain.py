"""Smoke test: build HiViT+UPerNet and load MIM pretrain checkpoint."""
import os
import sys
import os.path as osp
from collections import OrderedDict

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import mmcv_custom  # noqa: F401
import torch
from mmcv.utils import Config
from mmseg.models import build_segmentor
from mmcv_custom.checkpoint import _load_checkpoint

from backbone import hivit  # noqa: F401


def main():
    cfg_path = osp.join(
        PROJECT_ROOT,
        'configs/hivit/pixel_upernet_hivit_base_12_512_slide_160k_air_polarsar2_amp_linux.py')
    cfg = Config.fromfile(cfg_path)
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

    ckpt = _load_checkpoint(cfg.model.pretrained, map_location='cpu')
    sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
    enc_keys = {k for k in sd if k.startswith(('patch_embed', 'blocks', 'absolute_pos'))}
    dec_keys = {k for k in sd if k.startswith('decoder') or k in ('mask_token', 'norm.weight', 'norm.bias')}

    model.init_weights()
    loaded = model.backbone.state_dict()
    missing_enc = [k for k in loaded if k.split('.')[0] in ('patch_embed', 'blocks', 'absolute_pos_embed')
                   and k not in sd and 'fpn' not in k]
    print(f'checkpoint encoder keys: {len(enc_keys)}, decoder/extra keys: {len(dec_keys)}')
    print(f'missing encoder keys in ckpt: {len(missing_enc)}')
    if missing_enc:
        print('  sample:', missing_enc[:10])

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        feats = model.backbone(x)
    print('backbone out:', [tuple(f.shape) for f in feats])
    print('OK: HiViT pretrain loaded and forward pass succeeded.')


if __name__ == '__main__':
    main()
