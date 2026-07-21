# mmcv_custom_config/mmrotate/mmrotate/datasets/__init__.py
# Copyright (c) OpenMMLab. All rights reserved.
try:
    from .dior import DIORDataset  # noqa: F401, F403
except ImportError:
    DIORDataset = None

try:
    from .dota import DOTAv2Dataset  # noqa: F401, F403
    from .dota import DOTADataset, DOTAv15Dataset
except ImportError:
    DOTADataset = None
    DOTAv15Dataset = None
    DOTAv2Dataset = None

try:
    from .hrsc import HRSCDataset  # noqa: F401, F403
except ImportError:
    HRSCDataset = None

from .transforms import *  # noqa: F401, F403
from .dota_fruitnuts import DOTAFruitNutsDataset
from .dota_pipe import DOTAPipeDataset
from .dota_rgbd_pipe import DOTARGBDPipeDataset
from .collate import rgbd_collate_fn


__all__ = [
    "DOTAFruitNutsDataset",
    "DOTAPipeDataset",
    "DOTARGBDPipeDataset",
    'rgbd_collate_fn'
]

# Add available classes to __all__
if DOTADataset is not None:
    __all__.extend(["DOTADataset", "DOTAv15Dataset", "DOTAv2Dataset"])
if HRSCDataset is not None:
    __all__.append("HRSCDataset")
if DIORDataset is not None:
    __all__.append("DIORDataset")
