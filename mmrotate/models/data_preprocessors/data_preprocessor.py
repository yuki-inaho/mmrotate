# mmcv_custom_config/mmrotate/mmrotate/models/data_preprocessors/data_preprocessor.py
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor, stack_batch
from mmengine.logging import print_log
import logging

from mmengine.dataset import default_collate  # default_collate をインポート
from mmrotate.registry import MODELS  # mmrotateのMODELSを使用 (これが正しいか確認)
# from mmdet.registry import MODELS as MMDET_MODELS # あるいはMMDetのMODELSを使う場合


@MODELS.register_module()
class DetDataPreprocessorRGBD(BaseDataPreprocessor):
    """Image data pre-processor for RGB-D inputs in detection tasks.

    This class is a modification of mmdet.DetDataPreprocessor to handle
    4-channel RGB-D data, where the first 3 channels are RGB and the
    4th channel is Depth.

    It provides the following functionalities:

    - Collate and move data to the target device.
    - Automatically normalize images with specific mean and standard deviation.
      Separate normalization can be applied to RGB and Depth channels.
    - Pad inputs to the maximum size of current batch with defined fill values.
    - Convert inputs from bgr to rgb or rgb to bgr.
    - Batch augmentation.

    Args:
        rgb_mean (Sequence[Number], optional): The mean values (3 values)
            for RGB channels. Defaults to None.
        rgb_std (Sequence[Number], optional): The standard deviation values
            (3 values) for RGB channels. Defaults to None.
        depth_mean (Sequence[Number], optional): The mean value (1 value)
            for the Depth channel. Defaults to None.
        depth_std (Sequence[Number], optional): The standard deviation value
            (1 value) for the Depth channel. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded value for images.
            Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded value for semantic segmentation maps.
            Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to BGR.
            Defaults to False.
        boxtype2tensor (bool): Whether to convert BoxType to tensor during
            batch augmentation or data processing. Defaults to True.
        batch_augments (list[dict], optional): Batch-level augmentation
            configuration. Defaults to None.
    """

    def __init__(
        self,
        rgb_mean: Optional[Sequence[Number]] = None,
        rgb_std: Optional[Sequence[Number]] = None,
        depth_mean: Optional[Union[Number, Sequence[Number]]] = None,
        depth_std: Optional[Union[Number, Sequence[Number]]] = None,
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        boxtype2tensor: bool = True,  # MMEngine 0.6.0以降は True が一般的
        batch_augments: Optional[List[dict]] = None,
    ) -> None:
        super().__init__()
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor

        assert not (bgr_to_rgb and rgb_to_bgr), "`bgr_to_rgb` and `rgb_to_bgr` cannot be set to True at the same time"
        self.channel_conversion = bgr_to_rgb or rgb_to_bgr
        self.bgr_to_rgb = bgr_to_rgb
        self.rgb_to_bgr = rgb_to_bgr

        # RGB Normalization
        if rgb_mean is not None:
            assert rgb_std is not None, "rgb_std is None when rgb_mean is not None"
            assert len(rgb_mean) == 3, f"rgb_mean should have 3 values, but got {len(rgb_mean)}"
            assert len(rgb_std) == 3, f"rgb_std should have 3 values, but got {len(rgb_std)}"
            self.register_buffer("rgb_mean_buffer", torch.tensor(rgb_mean).view(-1, 1, 1), False)
            self.register_buffer("rgb_std_buffer", torch.tensor(rgb_std).view(-1, 1, 1), False)
            self.normalize_rgb = True
        else:
            self.normalize_rgb = False

        # Depth Normalization
        if depth_mean is not None:
            assert depth_std is not None, "depth_std is None when depth_mean is not None"
            if isinstance(depth_mean, Sequence):
                assert len(depth_mean) == 1, f"depth_mean sequence should have 1 value, but got {len(depth_mean)}"
                _depth_mean_val = depth_mean[0]
            else:  # single number
                _depth_mean_val = float(depth_mean)

            if isinstance(depth_std, Sequence):
                assert len(depth_std) == 1, f"depth_std sequence should have 1 value, but got {len(depth_std)}"
                _depth_std_val = depth_std[0]
            else:  # single number
                _depth_std_val = float(depth_std)

            self.register_buffer("depth_mean_buffer", torch.tensor([_depth_mean_val]).view(-1, 1, 1), False)
            self.register_buffer("depth_std_buffer", torch.tensor([_depth_std_val]).view(-1, 1, 1), False)
            self.normalize_depth = True
        else:
            self.normalize_depth = False

        if batch_augments is not None:
            # `batch_augments` は通常 MMDetection で定義されるため、MMDetのレジストリを使う
            from mmdet.registry import MODELS as MMDET_MODELS

            self.batch_augments = MMDET_MODELS.build(batch_augments)
        else:
            self.batch_augments = None

    def _get_pad_shape(self, data: dict) -> Tuple[int, int]:
        """Get the pad_shape of the batch based on data and pad_size_divisor."""
        data_samples = data.get("data_samples", None)
        inputs_tensor_list = data.get("inputs", None)  # `inputs` は通常リスト
        h, w = 0, 0

        if data_samples is not None and len(data_samples) > 0:
            # `data_samples` から形状を取得 (優先)
            all_h = [s.img_shape[0] for s in data_samples if s is not None and hasattr(s, "img_shape")]
            all_w = [s.img_shape[1] for s in data_samples if s is not None and hasattr(s, "img_shape")]
            if all_h and all_w:
                h = max(all_h)
                w = max(all_w)
            else:
                # data_samples に img_shape がない場合、inputs_tensor_list から取得試行
                print_log(
                    "Warning: No valid 'img_shape' in data_samples. Trying 'inputs'.",
                    logger="current",
                    level=logging.WARNING,
                )
                if (
                    isinstance(inputs_tensor_list, list)
                    and inputs_tensor_list
                    and isinstance(inputs_tensor_list[0], torch.Tensor)
                ):
                    h = max(inp.shape[-2] for inp in inputs_tensor_list)
                    w = max(inp.shape[-1] for inp in inputs_tensor_list)
                elif isinstance(inputs_tensor_list, torch.Tensor):  # バッチ済みテンソルの場合
                    h = inputs_tensor_list.shape[-2]
                    w = inputs_tensor_list.shape[-1]

        elif (
            isinstance(inputs_tensor_list, list)
            and inputs_tensor_list
            and isinstance(inputs_tensor_list[0], torch.Tensor)
        ):
            # data_samples がないが inputs_tensor_list がある場合
            print_log(
                "Warning: 'data_samples' not found. Using 'inputs' (list of tensors) shape for padding.",
                logger="current",
                level=logging.WARNING,
            )
            h = max(inp.shape[-2] for inp in inputs_tensor_list)
            w = max(inp.shape[-1] for inp in inputs_tensor_list)
        elif isinstance(inputs_tensor_list, torch.Tensor):  # バッチ済みテンソルの場合
            print_log(
                "Warning: 'data_samples' not found. Using 'inputs' (batched tensor) shape for padding.",
                logger="current",
                level=logging.WARNING,
            )
            h = inputs_tensor_list.shape[-2]
            w = inputs_tensor_list.shape[-1]

        if h == 0 or w == 0:
            print_log(
                "Error: Cannot determine target padding shape. Input data is insufficient.",
                logger="current",
                level=logging.ERROR,
            )
            # ここでエラーを発生させるか、デフォルト値を返す
            # 例として、もし inputs があればその形状を使う
            if isinstance(inputs_tensor_list, list) and inputs_tensor_list:
                return inputs_tensor_list[0].shape[-2:]
            elif isinstance(inputs_tensor_list, torch.Tensor):
                return inputs_tensor_list.shape[-2:]
            return (self.pad_size_divisor, self.pad_size_divisor)  # 最小のフォールバック

        if self.pad_size_divisor > 1:
            pad_h = (h + self.pad_size_divisor - 1) // self.pad_size_divisor * self.pad_size_divisor
            pad_w = (w + self.pad_size_divisor - 1) // self.pad_size_divisor * self.pad_size_divisor
        else:
            pad_h = h
            pad_w = w
        return pad_h, pad_w

    def forward(self, data: Union[dict, List[dict]], training: bool = False) -> Union[dict, list]:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.
        """
        if isinstance(data, list):
            try:
                data = default_collate(data)
            except Exception as e:
                print_log(
                    f"Error during default_collate: {e}. Data format might be unexpected.",
                    logger="current",
                    level=logging.ERROR,
                )
                if all(isinstance(d, dict) for d in data):
                    collated_data_simple = {k: [dic[k] for dic in data] for k in data[0]}
                    data = collated_data_simple
                else:
                    raise TypeError(
                        f"Data is a list, but elements are not all dicts, or default_collate failed. Got: {type(data[0])}"
                    )
        elif not isinstance(data, dict):
            raise TypeError(f"data should be a dict or a list of dict, but got {type(data)}")

        data = self.cast_data(data)

        inputs_tensor_list = data["inputs"]
        if not isinstance(inputs_tensor_list, list):
            if isinstance(inputs_tensor_list, torch.Tensor) and inputs_tensor_list.ndim == 4:
                inputs_tensor_list = list(inputs_tensor_list)
            else:
                raise TypeError(
                    f"inputs should be a list of Tensors or a 4D batched Tensor, got {type(inputs_tensor_list)}"
                )

        data_samples = data.get("data_samples", None)

        processed_inputs_list = []
        for img_tensor in inputs_tensor_list:
            if not isinstance(img_tensor, torch.Tensor):
                raise TypeError(f"Expected a torch.Tensor in inputs list, but got {type(img_tensor)}")

            target_device = img_tensor.device

            if img_tensor.ndim == 3 and img_tensor.shape[0] == 4:
                pass
            elif img_tensor.ndim == 4 and img_tensor.shape[0] == 1 and img_tensor.shape[1] == 4:
                img_tensor = img_tensor.squeeze(0)
            else:
                raise ValueError(
                    f"Input tensor for an item must be 3D (CHW, C=4) or 4D (BCHW, B=1, C=4), "
                    f"but got shape {img_tensor.shape}"
                )

            if not img_tensor.is_floating_point():
                img_float_tensor = img_tensor.float()
            else:
                img_float_tensor = img_tensor

            rgb_channels = img_float_tensor[:3, :, :]
            depth_channel = img_float_tensor[3:, :, :]

            if self.normalize_rgb:
                mean_rgb = self.rgb_mean_buffer.to(target_device)
                std_rgb = self.rgb_std_buffer.to(target_device)
                rgb_channels = (rgb_channels - mean_rgb) / std_rgb
            if self.normalize_depth:
                mean_d = self.depth_mean_buffer.to(target_device)
                std_d = self.depth_std_buffer.to(target_device)
                depth_channel = (depth_channel - mean_d) / std_d

            if self.channel_conversion:
                if self.bgr_to_rgb and not self.rgb_to_bgr:
                    rgb_channels = rgb_channels[[2, 1, 0], :, :]
                elif self.rgb_to_bgr and not self.bgr_to_rgb:
                    rgb_channels = rgb_channels[[2, 1, 0], :, :]

            processed_img_tensor = torch.cat((rgb_channels, depth_channel), dim=0)
            processed_inputs_list.append(processed_img_tensor)

        # Pad and stack the processed tensors in the list
        # `stack_batch` will pad to the largest image in the batch,
        # making dimensions divisible by `self.pad_size_divisor`.
        # The `_get_pad_shape` method and `pad_to_shape` argument are usually
        # not needed if BatchFixedSizePad (or equivalent in batch_augments)
        # has already ensured all images are of a consistent (or max) size before this point.
        # If `processed_inputs_list` contains tensors of varying H, W, `stack_batch` handles it.
        batch_inputs = stack_batch(
            processed_inputs_list,
            self.pad_size_divisor,
            self.pad_value,
            # pad_to_shape=batch_pad_shape_hw,  # ★ この行を削除またはコメントアウト
        )

        if data_samples is not None:
            actual_batch_input_shape_hw = tuple(batch_inputs.shape[-2:])
            for data_sample in data_samples:
                if data_sample is not None:
                    # `pad_shape` should reflect the shape *before* batching and stacking
                    # if it's different from `batch_input_shape`.
                    # However, mmdet's DetDataPreprocessor often sets pad_shape to batch_input_shape.
                    # For consistency, we can set both to the actual stacked shape.
                    data_sample.set_metainfo(
                        {
                            "batch_input_shape": actual_batch_input_shape_hw,
                            "pad_shape": actual_batch_input_shape_hw,
                        }
                    )
            if self.pad_mask and any(hasattr(ds, "gt_masks") for ds in data_samples if ds is not None):
                self.pad_gt_masks(data_samples, actual_batch_input_shape_hw)
            if self.pad_seg and any(hasattr(ds, "gt_sem_seg") for ds in data_samples if ds is not None):
                self.pad_gt_sem_seg(data_samples, actual_batch_input_shape_hw)

            if self.boxtype2tensor:
                for sample in data_samples:
                    if sample is not None:
                        if (
                            hasattr(sample, "gt_instances")
                            and hasattr(sample.gt_instances, "bboxes")
                            and not isinstance(sample.gt_instances.bboxes, torch.Tensor)
                        ):
                            try:
                                sample.gt_instances.bboxes = sample.gt_instances.bboxes.tensor
                            except AttributeError:
                                pass  # Might already be a tensor if converted by BaseBoxes
                        if (
                            hasattr(sample, "ignored_instances")
                            and hasattr(sample.ignored_instances, "bboxes")
                            and not isinstance(sample.ignored_instances.bboxes, torch.Tensor)
                        ):
                            try:
                                sample.ignored_instances.bboxes = sample.ignored_instances.bboxes.tensor
                            except AttributeError:
                                pass

        if training and self.batch_augments is not None:
            batch_inputs, data_samples = self.batch_augments(batch_inputs, data_samples)

        return {"inputs": batch_inputs, "data_samples": data_samples}

    def pad_gt_masks(self, data_samples: Sequence[Any], batch_pad_shape: Tuple[int, int]) -> None:
        """Pad `gt_masks` and stack into a batch tensor."""
        # data_samplesの各要素はDetDataSampleオブジェクトを期待
        if data_samples and hasattr(data_samples[0], "gt_masks"):
            for data_sample in data_samples:
                if data_sample is not None and hasattr(data_sample, "gt_masks") and data_sample.gt_masks is not None:
                    masks = data_sample.gt_masks.pad(batch_pad_shape, pad_val=self.mask_pad_value)
                    data_sample.gt_masks = masks

    def pad_gt_sem_seg(self, data_samples: Sequence[Any], batch_pad_shape: Tuple[int, int]) -> None:
        """Pad `gt_sem_seg` and stack into a batch tensor."""
        if data_samples and hasattr(data_samples[0], "gt_sem_seg"):
            for data_sample in data_samples:
                if (
                    data_sample is not None
                    and hasattr(data_sample, "gt_sem_seg")
                    and data_sample.gt_sem_seg is not None
                ):
                    gt_sem_seg = data_sample.gt_sem_seg.pad(batch_pad_shape, pad_val=self.seg_pad_value)
                    data_sample.gt_sem_seg = gt_sem_seg
