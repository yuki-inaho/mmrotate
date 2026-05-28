# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmdet
import mmengine
from mmengine.utils import digit_version

from .version import __version__, short_version

mmcv_minimum_version = '2.0.0rc4'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV {mmcv.__version__} is incompatible with MMRotate {__version__}. ' \
    f'Please use MMCV >= {mmcv_minimum_version}, ' \
    f'<= {mmcv_maximum_version} instead.'

mmengine_minimum_version = '0.6.0'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

mmdet_minimum_version = '3.0.0rc6'
# cu12 GPU validation (2026-05-28, fork branch cu121-mmrotate-migration-may28):
# Relax mmdet upper bound from 3.1.0 to 3.4.0 so we can pair mmrotate 1.0.0rc1
# with mmdet 3.3.0, which is the latest line that accepts mmcv 2.1.0 -- the
# lowest mmcv whose cu121 wheel is published. See workdoc
# temp/workdoc_May28-2026_cu12_submodule_gpu_validation.md §4.1.
mmdet_maximum_version = '3.4.0'
mmdet_version = digit_version(mmdet.__version__)

assert (mmdet_version >= digit_version(mmdet_minimum_version)
        and mmdet_version < digit_version(mmdet_maximum_version)), \
    f'MMDetection {mmdet.__version__} is incompatible ' \
    f'with MMRotate {__version__}. ' \
    f'Please use MMDetection >= {mmdet_minimum_version}, ' \
    f'< {mmdet_maximum_version} instead.'

__all__ = ['__version__', 'short_version', 'digit_version']
