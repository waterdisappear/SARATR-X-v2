from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class AIRPolSARSegDataset(CustomDataset):
    """AIR-PolSAR-Seg dataset with fixed 6-class taxonomy.

    With this dataset class, passing a subset in config `classes` will trigger
    mmseg's built-in label_map mechanism, so dropped classes are ignored in
    both training and evaluation.
    """

    CLASSES = (
        'housing',
        'industrial',
        'natural',
        'land_use',
        'water',
        'other',
    )
    PALETTE = [
        [255, 255, 0],    # housing
        [0, 0, 255],      # industrial
        [0, 255, 0],      # natural
        [255, 0, 0],      # land_use
        [0, 255, 255],    # water
        [255, 255, 255],  # other
    ]

    def __init__(self, **kwargs):
        super().__init__(reduce_zero_label=False, **kwargs)
