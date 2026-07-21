# Running this mmrotate fork on cu12 (torch 2.1.0 + cu121)

Part of the OpenMMLab cu12 migration (end goal: train
`tomato_pipe_rgbd_rtmdet_obb_training`, a Rotated RTMDet OBB model, on cu12).
This branch's base is upstream mmrotate **1.x** + a 1-line mmdet upper-bound guard
patch. Validated stack: **Python 3.10 / torch 2.1.0+cu121 / mmcv 2.2.0 /
mmengine 0.10.7 / mmdet 3.3.0 / mmrotate 1.0.0rc1**.

## Quick start (uv)

```bash
just sync          # uv sync (cu12 deps incl. mmcv 2.2.0 cu121 wheel, mmdet 3.3.0) + source on path (.pth)
just smoke         # import mmrotate (mmcv<=2.3.0 / mmdet<3.4.0 / mmengine<1.0 guards)
                   # + mmcv.ops.box_iou_rotated on GPU -> cuda_op_device: cuda
just env-doctor    # GPU + versions
```

`pyproject.toml` is a **virtual** uv project. The rotated CUDA op `box_iou_rotated`
comes from the mmcv 2.2.0 cu121 wheel. mmrotate is pure Python, installed via `.pth`.
