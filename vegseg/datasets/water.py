# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class WaterDataset(BaseSegDataset):
    """grass segmentation dataset. The file structure should be.

    .. code-block:: none

        ├── data
        │   ├── water
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├──xxx.png
        │   │   │   │   ├──...
        │   │   │   ├── val
        │   │   │   │   ├──xxxx.png
        │   │   │   │   ├──...
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├──xx.png
        │   │   │   │   ├──...
        │   │   │   ├── val
        │   │   │   │   ├──xxxxx.png
        │   │   │   │   ├──...
    """

    METAINFO = dict(
        classes=(
            "non-water",
            "main rivers",
            "small rivers",
            "lakes",
            "small water",
            "others water",
        ),
        palette=[
            [0, 0, 0],
            [0, 0, 255],
            [0, 128, 255],
            [0, 255, 255],
            [0, 255, 128],
            [0, 128, 128],
        ],
    )

    def __init__(
        self,
        img_suffix=".png",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )
