import argparse
import os
import os.path as osp
import sys

# 强制使用本仓库 mmseg_lwj 下的 mmcv_custom（自定义 PIPELINE 等）
PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
for _mod_name in list(sys.modules):
    if _mod_name == 'mmcv_custom' or _mod_name.startswith('mmcv_custom.'):
        del sys.modules[_mod_name]

import mmcv
import mmcv_custom  # noqa: F401
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from backbone import beit
from backbone import iTPN
from backbone import hivit  # noqa: F401

from mmcv_custom.rs_metrics import (
    accumulate_confusion_matrix,
    is_pre_eval_format,
    kappa_from_confusion,
    overall_accuracy_from_confusion,
)


def _print_air_polarsar_paper_metrics(dataset, outputs, eval_results):
    """论文常用指标：PA(OA)、mPA、mIoU、Kappa（后三项与 mmseg 表头 / 混淆矩阵一致）。"""
    if is_pre_eval_format(outputs):
        print(
            '\n[警告] 当前结果为 pre_eval 格式，无法汇总混淆矩阵计算 Kappa；'
            '需整图预测列表（与本仓库 ValTest hook 一致）。')
        return
    num_classes = len(dataset.CLASSES)
    cm = accumulate_confusion_matrix(
        outputs,
        dataset,
        num_classes=num_classes,
        ignore_index=getattr(dataset, 'ignore_index', 255),
        reduce_zero_label=getattr(dataset, 'reduce_zero_label', False),
        label_map=getattr(dataset, 'label_map', None) or {},
    )
    kappa = kappa_from_confusion(cm)
    pa = overall_accuracy_from_confusion(cm)
    mpa = eval_results.get('mAcc')
    miou = eval_results.get('mIoU')
    mpa_s = f'{float(mpa):.6f}' if mpa is not None else 'N/A'
    miou_s = f'{float(miou):.6f}' if miou is not None else 'N/A'
    aacc = eval_results.get('aAcc')
    pa_note = ''
    if aacc is not None:
        pa_note = '  (mmseg Summary 中 aAcc=%.6f，与 PA/OA 同义)' % float(aacc)
    print(
        '\n======== AIR-PolSAR-Seg 论文对齐指标 ========\n'
        '  PA / OA (整体像素精度，对角线之和/有效像素；与 Kappa 同源混淆矩阵): %.6f%s\n'
        '  mPA (各类别像素精度均值，对应 mmseg mAcc): %s\n'
        '  mIoU: %s\n'
        '  Kappa (多类 Kappa，见 mmcv_custom/rs_metrics.py): %.6f\n'
        '========================================' %
        (pa, pa_note, mpa_s, miou_s, kappa))


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_results = dataset.evaluate(outputs, args.eval, **kwargs)
            _print_air_polarsar_paper_metrics(dataset, outputs, eval_results)


if __name__ == '__main__':
    main()
