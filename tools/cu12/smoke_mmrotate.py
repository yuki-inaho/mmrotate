# Copyright (c) OpenMMLab. All rights reserved.
"""mmrotate cu12 GPU smoke.

Exercises a rotated CUDA op. Because mmrotate fork main is 0.x with an `mmcv-
full` dependency, this script attempts ``mmcv.ops.box_iou_rotated`` as the
rotated CUDA op (mmrotate >=1.x re-exports this from mmcv), and also records
the installed mmrotate version.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

from _smoke_common import run_smoke, standard_argv  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def runner() -> dict[str, Any]:
    import torch
    import mmrotate  # noqa: F401  (import side effect: ensures mmrotate loads)
    from mmcv.ops import box_iou_rotated

    boxes1 = torch.tensor(
        [[10.0, 10.0, 6.0, 4.0, 0.4], [25.0, 25.0, 8.0, 4.0, 0.0]],
        device='cuda',
        dtype=torch.float32,
    )
    boxes2 = torch.tensor(
        [[10.0, 10.0, 6.0, 4.0, 0.4], [22.0, 22.0, 8.0, 4.0, 0.2]],
        device='cuda',
        dtype=torch.float32,
    )
    iou = box_iou_rotated(boxes1, boxes2)
    torch.cuda.synchronize()
    return {
        'op': 'mmcv.ops.box_iou_rotated (rotated via mmcv, mmrotate imported)',
        'cuda_op_device': iou.device.type,
        'output_shape': list(iou.shape),
        'output_dtype': str(iou.dtype),
        'self_iou_top_left': float(iou[0, 0].item()),
    }


def main() -> int:
    args = standard_argv('mmrotate cu12 GPU smoke')
    return run_smoke(
        module_name='mmrotate', runner=runner, output_json=args.output_json)


if __name__ == '__main__':
    sys.exit(main())
