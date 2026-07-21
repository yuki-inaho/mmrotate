import copy
import inspect
from typing import Optional, Sequence, Tuple, Union, Dict, List, Any
import logging  # For print_log if logger 'current' is not setup by mmengine
import mmcv  # mmcv-full is expected
import numpy as np
import torch
import random  # ランダム処理のため追加

from mmengine.fileio import FileClient
from mmcv.image import imfrombytes  # mmcv.image.image から

# MMEngine and MMCV imports (respecting original structure)
from mmengine.logging import print_log  # Preferred logging
from mmengine.structures import InstanceData
from mmengine.utils import is_str

from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmcv.transforms import BaseTransform  # Base class for transforms

# MMDetection imports (respecting original structure)
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import HorizontalBoxes, BaseBoxes
from mmdet.datasets.transforms.transforms import (
    Resize,
    RandomFlip,
    Pad,
)  # Inheriting from MMDetection
from mmdet.datasets.transforms.formatting import to_tensor  # For modern data packing

# MMRotate imports (respecting original structure)
from mmrotate.structures.bbox import RotatedBoxes  # For OBBs
from mmrotate.registry import TRANSFORMS  # MMRotate's registry

import albumentations
from albumentations import Compose


@TRANSFORMS.register_module()
class ResizeRGBD(Resize):
    """Resize RGB images, depth maps, and rotated bounding boxes.

    This class inherits from `mmdet.datasets.transforms.Resize` and adds
    functionality to resize the depth map with a specific interpolation method.
    OBB resizing is handled by `_resize_bboxes` which is now aware of `RotatedBoxes`.
    """

    def __init__(self, depth_interpolation: str = "nearest", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_interpolation = depth_interpolation

    def _resize_depth(self, results: Dict) -> None:
        """Resize the depth map."""
        if "depth_map" in results and results.get("depth_map") is not None:
            # `results['scale']` は親クラスの_resize_imgで設定される目標サイズ (w, h)
            if "scale" not in results:
                print_log(
                    "'scale' key not found in results for depth resizing. Skipping depth resize.",
                    logger="current",
                    level=logging.WARNING,
                )
                return

            if results["scale"] == (
                results["depth_map"].shape[1],
                results["depth_map"].shape[0],
            ):
                return  # No need to resize

            resized_depth = mmcv.imresize(
                results["depth_map"],
                results["scale"],  # (w, h)
                interpolation=self.depth_interpolation,
                backend="cv2",
            )
            results["depth_map"] = resized_depth

    def transform(self, results: Dict) -> Optional[Dict]:
        """The transform function of `ResizeRGBD`.

        This function calls parent's `transform` to handle `img` and `gt_bboxes`
        (which now supports `RotatedBoxes` via `@autocast_box_type` in parent),
        and then resizes `depth_map`.
        """
        results = super().transform(results)

        if results is None:
            return None

        self._resize_depth(results)
        return results

    def __repr__(self) -> str:
        parent_repr = super().__repr__()
        return f"{parent_repr[:-1]}, depth_interpolation='{self.depth_interpolation}')"


@TRANSFORMS.register_module()
class RandomFlipRGBD(RandomFlip):
    """Flip RGB images, depth maps, and rotated bounding boxes.

    This class inherits from `mmdet.datasets.transforms.RandomFlip` and
    adds functionality to flip the depth map.
    Flipping of `RotatedBoxes` is handled by the `RotatedBoxes.flip` method,
    which is automatically called by the parent class's `_flip` method
    due to the `@autocast_box_type` decorator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _flip(self, results: Dict) -> None:
        """The flip function of `RandomFlipRGBD`.

        This function calls parent's `_flip` to handle `img`, `gt_bboxes`
        (including `RotatedBoxes`), `gt_masks`, etc., and then flips `depth_map`.
        """
        super()._flip(results)

        if results.get("flip"):
            if "depth_map" in results and results.get("depth_map") is not None:
                results["depth_map"] = mmcv.imflip(results["depth_map"], direction=results["flip_direction"])

    def __repr__(self) -> str:
        parent_repr = super().__repr__()
        return parent_repr.replace("RandomFlip", "RandomFlipRGBD", 1)


def normalize_angle_rad(angle_rad: float, period: float = 2 * np.pi, offset: float = -np.pi) -> float:
    """Normalize angle to a specific range, e.g., (-pi, pi]."""
    return (angle_rad - offset) % period + offset


@TRANSFORMS.register_module()
class LoadRGBDImageFromFile(BaseTransform):
    """Load an RGB image and its corresponding depth map from file.
    Outputs 'img' (3-channel, typically BGR from OpenCV backend) and
    'depth_map' (1-channel depth) separately by default.
    Compatible with mmcv==2.0.0.
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "color",
        imdecode_backend: str = "cv2",
        backend_args: Optional[dict] = None,
        depth_to_meters_scale: float = 0.001,
        depth_threshold_min_mm: float = 0.0,
        depth_threshold_max_mm: float = 1500.0,
        concatenate_rgbd: bool = False,
        clip_depth_to_max: bool = False,
        invalid_depth_value: float = 0.0,
        ignore_empty_depth: bool = False,
        file_client_args: Optional[dict] = None,
    ):
        super().__init__()
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.depth_to_meters_scale = depth_to_meters_scale

        self._depth_threshold_min_mm_config = depth_threshold_min_mm
        self._depth_threshold_max_mm_config = depth_threshold_max_mm
        self.depth_threshold_min_m = depth_threshold_min_mm * self.depth_to_meters_scale
        self.depth_threshold_max_m = depth_threshold_max_mm * self.depth_to_meters_scale
        self.concatenate_rgbd = concatenate_rgbd
        self.clip_depth_to_max = clip_depth_to_max
        self.invalid_depth_value_m = invalid_depth_value
        self.ignore_empty_depth = ignore_empty_depth

        if file_client_args is not None:
            print_log(
                "Warning: 'file_client_args' is deprecated, please use 'backend_args'.",
                logger="current",
                level=logging.WARNING,
            )
            if backend_args is not None:
                raise ValueError("Cannot set both 'file_client_args' and 'backend_args'.")
            self.backend_args = file_client_args.copy()
        elif backend_args is not None:
            self.backend_args = backend_args.copy()
        else:
            self.backend_args = dict(backend="disk")

        self.file_client = None

    def _load_image_with_file_client(self, filepath: str, flag: str) -> Optional[np.ndarray]:
        if self.file_client is None:
            self.file_client = FileClient(**self.backend_args)

        try:
            img_bytes = self.file_client.get(
                filepath
            )  # This will raise FileNotFoundError if file doesn't exist with disk backend
            img = imfrombytes(img_bytes, flag=flag, backend=self.imdecode_backend)
            if img is None:
                print_log(
                    f"imfrombytes returned None for: {filepath} (flag: {flag})",
                    logger="current",
                    level=logging.ERROR,
                )
            return img
        except FileNotFoundError:  # Catch FileNotFoundError explicitly
            print_log(
                f"File not found by FileClient for: {filepath}",
                logger="current",
                level=logging.ERROR,
            )
            return None
        except Exception as e:
            # import traceback # Not needed if exc_info=True is used in print_log
            print_log(
                f"Exception in _load_image_with_file_client for {filepath}: {e}",
                logger="current",
                level=logging.ERROR,
                exc_info=True,
            )
            return None

    def transform(self, results: Dict) -> Optional[Dict]:
        rgb_filename = results.get("img_path")
        depth_filename = results.get("depth_map_path")

        if not rgb_filename or not is_str(rgb_filename):
            raise ValueError(f"'img_path' must be a valid string, but got {rgb_filename}")
        if not depth_filename or not is_str(depth_filename):
            raise ValueError(f"'depth_map_path' must be a valid string, but got {depth_filename}")

        img = self._load_image_with_file_client(rgb_filename, self.color_type)
        if img is None:
            return None  # Error logged in helper

        if img.ndim == 2:
            print_log(
                f"RGB image {rgb_filename} is grayscale, converting to BGR.",
                logger="current",
                level=logging.WARNING,
            )
            img = mmcv.image.gray2bgr(img)  # Use mmcv.image explicitly
        elif img.ndim == 3 and img.shape[2] == 4:
            print_log(
                f"RGB image {rgb_filename} is RGBA, converting to BGR.",
                logger="current",
                level=logging.WARNING,
            )
            img = mmcv.image.rgba2bgr(img)
        elif not (img.ndim == 3 and img.shape[2] == 3):
            print_log(
                f"Loaded RGB image {rgb_filename} has unexpected shape: {img.shape}. Could be an issue.",
                logger="current",
                level=logging.ERROR,
            )
            return None  # Treat as critical error

        depth_map_raw = self._load_image_with_file_client(depth_filename, "unchanged")
        if depth_map_raw is None:
            return None

        if depth_map_raw.ndim == 3:
            if depth_map_raw.shape[2] == 1:
                depth_map = depth_map_raw[:, :, 0]
            elif depth_map_raw.shape[2] >= 3:
                print_log(
                    f"Depth map {depth_filename} has {depth_map_raw.shape[2]} channels. Converting to grayscale.",
                    logger="current",
                    level=logging.WARNING,
                )
                depth_map = mmcv.image.rgb2gray(depth_map_raw, keep_alpha=False)
            else:
                raise ValueError(f"Depth map {depth_filename} has unsupported 3D shape {depth_map_raw.shape}.")
        elif depth_map_raw.ndim == 2:
            depth_map = depth_map_raw
        else:
            raise ValueError(f"Depth map {depth_filename} has unsupported ndim {depth_map_raw.ndim}.")

        ori_rgb_shape = img.shape[:2]
        ori_depth_shape_actual = depth_map.shape[:2]

        depth_map_m = depth_map.astype(np.float32) * self.depth_to_meters_scale

        nan_mask = np.isnan(depth_map_m)
        depth_map_m[nan_mask] = self.invalid_depth_value_m

        valid_mask = (depth_map_m >= self.depth_threshold_min_m) & (depth_map_m <= self.depth_threshold_max_m)

        depth_map_processed = depth_map_m.copy()
        if self.clip_depth_to_max:
            depth_map_processed[depth_map_processed > self.depth_threshold_max_m] = self.depth_threshold_max_m
            valid_mask = (depth_map_processed >= self.depth_threshold_min_m) & (
                depth_map_processed <= self.depth_threshold_max_m
            )

        depth_map_processed[~valid_mask] = self.invalid_depth_value_m

        if not self.ignore_empty_depth and np.all(depth_map_processed == self.invalid_depth_value_m):
            print_log(
                f"All depth values are invalid in {depth_filename} after processing. Skipping.",
                logger="current",
                level=logging.WARNING,
            )
            return None

        if self.to_float32 and not self.concatenate_rgbd:
            img = img.astype(np.float32)

        results["img"] = img
        results["img_shape"] = ori_rgb_shape
        results["ori_shape"] = ori_rgb_shape
        results["num_rgb_channels"] = results["img"].shape[2] if results["img"].ndim == 3 else 1

        results["pad_shape"] = ori_rgb_shape
        results["scale_factor"] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        results.setdefault("img_fields", []).append("img")

        if self.concatenate_rgbd:
            if results["img"].shape[:2] != depth_map_processed.shape[:2]:
                print_log(
                    f"RGB {results['img'].shape[:2]} and Depth {depth_map_processed.shape[:2]} spatial dimensions "
                    f"mismatch for concat. Resizing depth to match RGB using nearest interpolation.",
                    logger="current",
                    level=logging.WARNING,
                )
                depth_map_processed = mmcv.imresize(  # mmcv.image.imresize or mmcv.imresize
                    depth_map_processed,
                    (results["img"].shape[1], results["img"].shape[0]),
                    interpolation="nearest",
                )

            img_for_concat = results["img"].astype(np.float32) if results["img"].dtype != np.float32 else results["img"]
            depth_to_concat = depth_map_processed[:, :, np.newaxis].astype(np.float32)

            results["img"] = np.concatenate((img_for_concat, depth_to_concat), axis=2)
            results["img_shape"] = results["img"].shape[:2]
            results["num_depth_channels"] = depth_to_concat.shape[2]

            results.pop("depth_map", None)
            results.pop("depth_map_shape", None)
            results.pop("ori_depth_shape", None)
            results["depth_fields"] = []
        else:
            results["depth_map"] = depth_map_processed
            results["ori_depth_shape"] = ori_depth_shape_actual
            results["depth_map_shape"] = depth_map_processed.shape[:2]
            results["num_depth_channels"] = 1
            results.setdefault("depth_fields", []).append("depth_map")

        return results

    def __repr__(self) -> str:
        depth_min_mm_repr = self._depth_threshold_min_mm_config
        depth_max_mm_repr = self._depth_threshold_max_mm_config

        return (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"imdecode_backend='{self.imdecode_backend}', "
            f"backend_args={self.backend_args}, "
            f"depth_to_meters_scale={self.depth_to_meters_scale}, "
            f"depth_threshold_min_mm={depth_min_mm_repr}, "
            f"depth_threshold_max_mm={depth_max_mm_repr}, "
            f"concatenate_rgbd={self.concatenate_rgbd}, "
            f"clip_depth_to_max={self.clip_depth_to_max}, "
            f"invalid_depth_value(m)={self.invalid_depth_value_m}, "
            f"ignore_empty_depth={self.ignore_empty_depth})"
        )


@TRANSFORMS.register_module()
class PadRGBD(Pad):  # Inherit from mmdet.datasets.transforms.Pad
    """Pad RGB image and depth map to a specific size or multiple of size_divisor.
    Extends mmdet.datasets.transforms.Pad.

    This transform pads:
    - 'img' (RGB or concatenated RGBD) using `pad_val_rgb`. The parent class
      handles this based on the `pad_val` dict passed to its `__init__`.
    - 'depth_map' (if it exists as a separate field and is listed in `depth_fields`)
      using a specific `pad_val_depth`.

    Args:
        size (tuple, optional): Fixed padding target size (w, h) or (h,w) depending
            on parent Pad convention. MMDET Pad usually expects (h,w) for `size`
            if `pad_to_square` is False when `size` is a tuple of 2 ints.
            If `size` is an int, it pads to a square of that size.
            Config files often use (w,h) for image scales. This class passes `size`
            directly to parent, so ensure consistency with parent `Pad`'s expectation.
        size_divisor (int, optional): The divisor of padded image size.
            Pads the image so that each side is divisible by `size_divisor`.
        pad_to_square (bool): Whether to pad the image into a square. If True,
            the `size` is calculated as the max of H and W. Defaults to False.
        pad_val_rgb (Union[float, Sequence[float]]): Value to use for padding 'img' field.
            If 'img' is RGB (3-ch), this should be float or 3-tuple (e.g., BGR order).
            If 'img' is RGBD (e.g., 4-ch from LoadRGBDImageFromFile(concatenate_rgbd=True)),
            this should be float (broadcasted) or a tuple of length matching img channels.
            MMCV's impad handles broadcasting of float or matching tuple length.
            Defaults to 0.0.
        pad_val_depth (float): Value to pad the separate 'depth_map' field. Defaults to 0.0.
        padding_mode (str): Type of padding. Defaults to 'constant'.
    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int]]] = None,  # Parent Pad can take int for square
        size_divisor: Optional[int] = None,
        pad_to_square: bool = False,
        # Argument name changed to match the common calling convention from config files
        pad_val_rgb: Union[float, Sequence[float]] = 0.0,
        pad_val_depth: float = 0.0,
        padding_mode: str = "constant",
        # Other potential parent Pad arguments (check mmdet.Pad source for your version):
        # pad_to_square_edge (str, optional): Which side to pad to square. Defaults to 'bottom_right'.
        #                       Only valid if `pad_to_square=True`.
        # pad_value_divisible (bool): Whether `pad_val` should be divisible by
        #                       `size_divisor`. Defaults to False.
        # If parent `Pad` takes `**kwargs` and passes them to `BaseTransform`,
        # it's safer to not include `**kwargs` here unless specifically forwarding.
        # MMDET Pad __init__ does not usually take **kwargs beyond its defined args.
    ):
        # The parent Pad class (mmdet.Pad) expects `pad_val` to be a dict
        # mapping field names (like 'img', 'gt_masks') to their padding values.
        parent_pad_val_dict = dict(img=pad_val_rgb)
        # If you also need to pad segmentation masks with a specific value:
        # parent_pad_val_dict['gt_masks'] = your_mask_pad_value # e.g., 0
        # parent_pad_val_dict['gt_seg'] = your_seg_pad_value   # e.g., 255 (ignore_label)

        super().__init__(
            size=size,
            size_divisor=size_divisor,
            pad_to_square=pad_to_square,
            pad_val=parent_pad_val_dict,  # Pass the constructed dict for parent to use
            padding_mode=padding_mode,
            # If parent Pad takes more specific args like pad_to_square_edge, ensure they are passed
        )

        self.pad_val_depth_config = float(pad_val_depth)
        # Store the original config value for 'img's padding for __repr__ or other logic
        self.pad_val_rgb_config = pad_val_rgb

    def _pad_depth(self, results: Dict[str, Any]) -> None:
        """Pad 'depth_map' field using self.pad_val_depth_config."""
        if "depth_map" in results.get("depth_fields", []) and results.get("depth_map") is not None:
            if "padding" not in results:
                ori_h, ori_w = results["depth_map"].shape[:2]
                current_size_arg = self.size

                if self.pad_to_square:
                    target_h = target_w = max(ori_h, ori_w)
                elif self.size_divisor is not None:
                    target_h = (ori_h + self.size_divisor - 1) // self.size_divisor * self.size_divisor
                    target_w = (ori_w + self.size_divisor - 1) // self.size_divisor * self.size_divisor
                elif current_size_arg is not None:
                    if isinstance(current_size_arg, int):  # Pad to square of this size
                        target_h = target_w = current_size_arg
                    elif isinstance(current_size_arg, tuple) and len(current_size_arg) == 2:
                        target_w_cfg, target_h_cfg = (
                            current_size_arg[0],
                            current_size_arg[1],
                        )
                        target_h, target_w = target_h_cfg, target_w_cfg
                    else:
                        print_log(
                            "PadRGBD: Cannot determine target size for depth padding, 'size' is invalid.",
                            logger="current",
                            level=logging.ERROR,
                        )
                        return
                else:
                    print_log(
                        "PadRGBD: No valid padding rule found for depth (padding key missing & no size info).",
                        logger="current",
                        level=logging.ERROR,
                    )
                    return

                pad_h_bottom = max(0, target_h - ori_h)
                pad_w_right = max(0, target_w - ori_w)
                current_padding = (0, pad_h_bottom, 0, pad_w_right)
                print_log(
                    f"PadRGBD._pad_depth: 'padding' key not found. Calculated padding for depth: {current_padding} "
                    f"(ori_shape: {(ori_h, ori_w)}, target_shape: {(target_h, target_w)})",
                    logger="current",
                    level=logging.DEBUG,
                )
            else:
                current_padding = results["padding"]

            padded_depth = mmcv.impad(
                results["depth_map"],
                padding=current_padding,
                pad_val=self.pad_val_depth_config,
                padding_mode=self.padding_mode,
            )
            results["depth_map"] = padded_depth
            if "depth_map_shape" in results:  # Update if key exists
                results["depth_map_shape"] = padded_depth.shape[:2]

    def transform(self, results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        results_after_super = super().transform(results)
        if results_after_super is None:
            return None
        self._pad_depth(results_after_super)

        # Ensure pad_shape only contains spatial dimensions (H, W)
        if "pad_shape" in results_after_super:
            pad_shape = results_after_super["pad_shape"]
            if len(pad_shape) > 2:
                results_after_super["pad_shape"] = pad_shape[:2]  # Only (H, W)

        return results_after_super

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        if hasattr(self, "size") and self.size is not None:
            s += f"size={self.size}, "
        if hasattr(self, "size_divisor") and self.size_divisor is not None:
            s += f"size_divisor={self.size_divisor}, "
        if hasattr(self, "pad_to_square") and self.pad_to_square:
            s += f"pad_to_square={self.pad_to_square}, "
        s += f"pad_val_rgb={self.pad_val_rgb_config}, "
        s += f"pad_val_depth={self.pad_val_depth_config}, "
        s += f"padding_mode='{self.padding_mode}'"
        s += ")"
        return s


@TRANSFORMS.register_module()
class NormalizeRGBD(BaseTransform):
    """Normalize the RGB and Depth channels of an image.

    Added key is "img_norm_cfg".
    Required Keys:
        - img (np.ndarray): Image to be normalized. Can be RGB (H,W,C_rgb) or
                            RGBD concatenated (H,W,C_rgb+C_depth).
        - depth_map (np.ndarray, optional): Separate depth map (H,W) if not concatenated.
        - num_rgb_channels (int): Number of RGB channels in 'img'.
        - num_depth_channels (int): Number of depth channels (in 'img' if concatenated, or from 'depth_map').

    Modified Keys:
        - img
        - depth_map (if separate and processed)

    Args:
        mean (Sequence[float]): Mean values for RGB channels.
        std (Sequence[float]): Standard deviations for RGB channels.
        depth_mean (Sequence[float]): Mean values for Depth channel(s).
        depth_std (Sequence[float]): Standard deviations for Depth channel(s).
        to_rgb (bool): Whether to convert image from BGR to RGB before normalization.
            Applied only to the RGB part. Defaults to True.
            Set to False if input RGB part is already in RGB format.
        concatenate_input (bool): If True, 'img' is assumed to be concatenated RGBD.
                                  If False, 'img' is RGB and 'depth_map' is separate Depth.
                                  This flag is read from `results` if available, else uses init arg.
                                  Defaults to False (expecting separate img and depth_map).
    """

    def __init__(
        self,
        mean: List[float],
        std: List[float],
        depth_mean: List[float],
        depth_std: List[float],
        to_rgb: bool = True,
        assume_concatenated_input: bool = False,
    ):  # Init arg to guide behavior if 'concatenate_input' not in results
        super().__init__()
        num_expected_rgb_ch = 3  # Common case
        if not (len(mean) == len(std) == num_expected_rgb_ch):
            print_log(
                f"Warning: NormalizeRGBD expects mean/std for {num_expected_rgb_ch} RGB channels, "
                f"but got {len(mean)} for mean and {len(std)} for std. Ensure this is intended.",
                logger="current",
                level=logging.WARNING,
            )

        if not (len(depth_mean) == len(depth_std)):
            raise ValueError("Depth Mean and Std must have the same number of values.")

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.depth_mean = np.array(depth_mean, dtype=np.float32)
        self.depth_std = np.array(depth_std, dtype=np.float32)
        self.to_rgb_flag = to_rgb
        self.assume_concatenated_input = assume_concatenated_input

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"].astype(np.float32)
        num_rgb_channels = results.get("num_rgb_channels", self.mean.shape[0])
        is_concatenated = results.get("concatenate_input", self.assume_concatenated_input)

        cfg_mean_rgb = self.mean.tolist()
        cfg_std_rgb = self.std.tolist()
        cfg_depth_mean = self.depth_mean.tolist()
        cfg_depth_std = self.depth_std.tolist()
        current_mean_rgb = self.mean[:num_rgb_channels]
        current_std_rgb = self.std[:num_rgb_channels]

        if is_concatenated:
            if img.shape[2] < num_rgb_channels + len(self.depth_mean):
                raise ValueError(
                    f"Concatenated image has {img.shape[2]} channels, but expected at least "
                    f"{num_rgb_channels} (RGB) + {len(self.depth_mean)} (Depth) channels."
                )

            rgb_part = img[..., :num_rgb_channels]
            depth_part = img[..., num_rgb_channels : num_rgb_channels + len(self.depth_mean)]

            rgb_part_normalized = mmcv.imnormalize(rgb_part, current_mean_rgb, current_std_rgb, self.to_rgb_flag)
            depth_part_normalized = mmcv.imnormalize(depth_part, self.depth_mean, self.depth_std, to_rgb=False)

            if img.shape[2] > num_rgb_channels + len(self.depth_mean):
                extra_channels = img[..., num_rgb_channels + len(self.depth_mean) :]
                results["img"] = np.concatenate(
                    (rgb_part_normalized, depth_part_normalized, extra_channels),
                    axis=-1,
                )
            else:
                results["img"] = np.concatenate((rgb_part_normalized, depth_part_normalized), axis=-1)
        else:
            # 'img' is RGB, 'depth_map' is separate
            results["img"] = mmcv.imnormalize(img, current_mean_rgb, current_std_rgb, self.to_rgb_flag)

            if "depth_map" in results.get("depth_fields", []) and results["depth_map"] is not None:
                depth_map = results["depth_map"].astype(np.float32)

                if depth_map.ndim == 2:
                    depth_map_for_norm = depth_map[..., np.newaxis]
                elif depth_map.ndim == 3 and depth_map.shape[2] == len(self.depth_mean):
                    depth_map_for_norm = depth_map
                else:
                    raise ValueError(
                        f"Depth map shape {depth_map.shape} incompatible with depth_mean length {len(self.depth_mean)}"
                    )

                normalized_depth = mmcv.imnormalize(depth_map_for_norm, self.depth_mean, self.depth_std, to_rgb=False)

                results["depth_map"] = normalized_depth[..., 0] if depth_map.ndim == 2 else normalized_depth
            elif "depth_map" in results.get("depth_fields", []) and results["depth_map"] is None:
                print_log(
                    "NormalizeRGBD: 'depth_map' is None, skipping its normalization.",
                    logger="current",
                    level=logging.DEBUG,
                )

        results["img_norm_cfg"] = dict(
            mean=cfg_mean_rgb,
            std=cfg_std_rgb,
            depth_mean=cfg_depth_mean,
            depth_std=cfg_depth_std,
            to_rgb=self.to_rgb_flag,
            concatenated_input_during_norm=is_concatenated,
        )
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"mean={self.mean.tolist()}, std={self.std.tolist()}, "
            f"depth_mean={self.depth_mean.tolist()}, depth_std={self.depth_std.tolist()}, "
            f"to_rgb={self.to_rgb_flag}, assume_concatenated_input={self.assume_concatenated_input})"
        )


@TRANSFORMS.register_module()
class PackRGBDDetInputs(BaseTransform):
    """Pack the inputs data for rotated detection using RGB-D.

    This transform is typically the last step in the data pipeline.
    It performs the following steps:
    1. Concatenates 'img' (RGB) and 'depth_map' into a 4-channel tensor.
    2. Converts the 4-channel tensor to PyTorch tensor.
    3. Packs ground truth annotations (bboxes, labels, masks, ignore_flags)
       into a DetDataSample structure, separating gt_instances and
       ignored_instances.
    4. Ensures that `ignored_instances` always has a 'bboxes' key, even if empty,
       for compatibility with evaluators like DOTAMetric.

    Required Keys in `results`:
        - img (np.ndarray): The RGB image (H, W, 3).
        - depth_map (np.ndarray): The depth map (H, W) or (H, W, 1).
        - gt_bboxes (BaseBoxes or convertible): Ground truth bounding boxes.
        - gt_bboxes_labels (np.ndarray or list): Ground truth labels.
        - gt_ignore_flags (np.ndarray or list, optional): Flags indicating
          whether an instance should be ignored.
        - gt_masks (BitmapMasks | PolygonMasks, optional): Ground truth masks.
        - (meta_keys specified in __init__)

    Modified Keys in `packed_results`:
        - inputs (torch.Tensor): The 4-channel RGB-D tensor (4, H, W).
        - data_samples (DetDataSample): Containing gt_instances,
          ignored_instances, and metainfo.
    """

    def __init__(
        self,
        meta_keys: Sequence[str] = (
            "img_id",
            "img_path",
            "depth_map_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "homography_matrix",
            "ori_depth_shape",
            "depth_map_shape",
            # "pad_shape", # Usually added by DataPreprocessor, but can be included if known
        ),
        pack_ignored_instances_flag: bool = True,  # Renamed for clarity
    ):
        self.meta_keys = meta_keys
        self.pack_ignored_instances_flag = pack_ignored_instances_flag

    def _ensure_baseboxes(self, bboxes_data):
        """Ensures the input is a BaseBoxes instance, converting if necessary."""
        if isinstance(bboxes_data, BaseBoxes):
            return bboxes_data
        if isinstance(bboxes_data, (np.ndarray, torch.Tensor, list)):
            try:
                # Try to infer type or default to RotatedBoxes for mmrotate
                return RotatedBoxes(bboxes_data)
            except Exception as e:
                raise TypeError(
                    f"Cannot convert bboxes of type {type(bboxes_data)} to BaseBoxes (tried RotatedBoxes). Error: {e}"
                )
        raise TypeError(f"Unsupported bbox data type: {type(bboxes_data)}")

    def transform(self, results: Dict) -> Dict:
        packed_results = dict()

        # --- 1. Prepare 4-channel input tensor ---
        if "img" not in results:
            raise ValueError("'img' (RGB) must be present in results.")
        if "depth_map" not in results:
            raise ValueError("'depth_map' must be present in results.")

        img_np = results["img"]
        depth_map_np = results["depth_map"]

        if not (img_np.ndim == 3 and img_np.shape[-1] == 3):
            raise ValueError(f"Input 'img' must be HWC with 3 channels, got {img_np.shape}")
        if not (depth_map_np.ndim == 2 or (depth_map_np.ndim == 3 and depth_map_np.shape[-1] == 1)):
            raise ValueError(f"Input 'depth_map' must be HW or HWC with 1 channel, got {depth_map_np.shape}")

        if depth_map_np.ndim == 2:
            depth_map_np = np.expand_dims(depth_map_np, axis=-1)

        if img_np.shape[:2] != depth_map_np.shape[:2]:
            raise ValueError(
                f"Spatial dimensions of RGB image {img_np.shape[:2]} "
                f"and depth_map {depth_map_np.shape[:2]} do not match. "
                "Check preceding transforms like ResizeRGBD or PadRGBD."
            )

        # Ensure float32 for concatenation if needed by model/preprocessor
        # Normalization should occur in DataPreprocessor or a Normalize transform.
        # Here, we just ensure types are consistent for concatenation.
        img_for_concat = img_np.astype(np.float32) if img_np.dtype != np.float32 else img_np
        depth_for_concat = depth_map_np.astype(np.float32) if depth_map_np.dtype != np.float32 else depth_map_np

        inputs_np = np.concatenate((img_for_concat, depth_for_concat), axis=2)  # (H, W, 4)
        packed_results["inputs"] = to_tensor(inputs_np.transpose(2, 0, 1))  # (4, H, W)

        # --- 2. Pack annotations into DetDataSample ---
        data_sample = DetDataSample()

        # Ground Truth Instances (non-ignored)
        gt_instances_data = InstanceData()

        original_gt_bboxes = None
        original_gt_labels_tensor = None
        original_gt_masks = None

        if "gt_bboxes" in results and results["gt_bboxes"] is not None:
            original_gt_bboxes = self._ensure_baseboxes(results["gt_bboxes"])
            if len(original_gt_bboxes) > 0:
                if "gt_bboxes_labels" in results and results["gt_bboxes_labels"] is not None:
                    original_gt_labels_tensor = to_tensor(results["gt_bboxes_labels"])
                    if len(original_gt_bboxes) != len(original_gt_labels_tensor):
                        min_len = min(len(original_gt_bboxes), len(original_gt_labels_tensor))
                        print_log(
                            f"Warning: Mismatch num gt_bboxes ({len(original_gt_bboxes)}) and "
                            f"gt_bboxes_labels ({len(original_gt_labels_tensor)}). Truncating to {min_len}.",
                            logger="current",
                            level=logging.WARNING,
                        )
                        original_gt_bboxes = original_gt_bboxes[:min_len]
                        original_gt_labels_tensor = original_gt_labels_tensor[:min_len]
                else:
                    print_log(
                        "Warning: 'gt_bboxes' present but 'gt_bboxes_labels' is missing.",
                        logger="current",
                        level=logging.WARNING,
                    )

                if "gt_masks" in results and results["gt_masks"] is not None:
                    original_gt_masks = results["gt_masks"]
                    if len(original_gt_bboxes) != len(original_gt_masks):
                        print_log(
                            f"Warning: Mismatch num gt_bboxes ({len(original_gt_bboxes)}) and "
                            f"gt_masks ({len(original_gt_masks)}). Masks might not correspond correctly.",
                            logger="current",
                            level=logging.WARNING,
                        )
                        original_gt_masks = original_gt_masks[: len(original_gt_bboxes)]

        # Determine valid (non-ignored) and ignored indices
        valid_idx = None
        ignore_idx = None

        if original_gt_bboxes is not None and len(original_gt_bboxes) > 0:
            if "gt_ignore_flags" in results and results["gt_ignore_flags"] is not None:
                ignore_flags_tensor = torch.tensor(results["gt_ignore_flags"], dtype=torch.bool)
                if len(ignore_flags_tensor) == len(original_gt_bboxes):
                    valid_idx = ~ignore_flags_tensor
                    ignore_idx = ignore_flags_tensor
                else:
                    print_log(
                        f"Warning: Mismatch num gt_bboxes ({len(original_gt_bboxes)}) and "
                        f"gt_ignore_flags ({len(ignore_flags_tensor)}). Assuming all instances are valid for gt_instances.",
                        logger="current",
                        level=logging.WARNING,
                    )
                    valid_idx = torch.ones(
                        len(original_gt_bboxes),
                        dtype=torch.bool,
                        device=original_gt_bboxes.device,
                    )
            else:
                valid_idx = torch.ones(
                    len(original_gt_bboxes),
                    dtype=torch.bool,
                    device=original_gt_bboxes.device,
                )

            if valid_idx is not None and torch.any(valid_idx):
                gt_instances_data.bboxes = original_gt_bboxes[valid_idx]
                if original_gt_labels_tensor is not None:
                    gt_instances_data.labels = original_gt_labels_tensor[valid_idx]
                if original_gt_masks is not None and len(original_gt_masks) == len(original_gt_bboxes):
                    gt_instances_data.masks = original_gt_masks[valid_idx.cpu().numpy()]

        data_sample.gt_instances = gt_instances_data

        # Ignored Instances
        ignored_instances_data = InstanceData()
        if (
            self.pack_ignored_instances_flag
            and original_gt_bboxes is not None
            and len(original_gt_bboxes) > 0
            and ignore_idx is not None
            and torch.any(ignore_idx)
        ):
            ignored_instances_data.bboxes = original_gt_bboxes[ignore_idx]
            if original_gt_labels_tensor is not None:
                ignored_instances_data.labels = original_gt_labels_tensor[ignore_idx]
            if original_gt_masks is not None and len(original_gt_masks) == len(original_gt_bboxes):
                ignored_instances_data.masks = original_gt_masks[ignore_idx.cpu().numpy()]
        else:
            ref_bbox_type = RotatedBoxes
            if original_gt_bboxes is not None and len(original_gt_bboxes) > 0:
                ref_bbox_type = type(original_gt_bboxes)

            dim = 5
            if hasattr(ref_bbox_type, "empty_tensor_dim"):
                dim = ref_bbox_type.empty_tensor_dim
            elif original_gt_bboxes is not None and original_gt_bboxes.tensor.numel() > 0:
                dim = original_gt_bboxes.size(-1)

            ignored_instances_data.bboxes = ref_bbox_type(torch.empty(0, dim, device=packed_results["inputs"].device))
            if original_gt_labels_tensor is not None:
                ignored_instances_data.labels = torch.empty(
                    0,
                    dtype=original_gt_labels_tensor.dtype,
                    device=packed_results["inputs"].device,
                )

        data_sample.ignored_instances = ignored_instances_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample
        return packed_results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(meta_keys={self.meta_keys}, pack_ignored_instances_flag={self.pack_ignored_instances_flag})"


@TRANSFORMS.register_module()
class GenerateNullDepthImageFromRGB(BaseTransform):
    """Generate a null depth map from RGB image dimensions."""

    def __init__(self, depth_fill_value: float = 0.0) -> None:
        self.depth_fill_value = depth_fill_value

    def transform(self, results: dict) -> dict:
        if "img" not in results:
            raise KeyError("Key 'img' is missing. LoadImageFromFile should be called first.")
        img_shape_hw = results["img"].shape[:2]
        null_depth_map = np.full(img_shape_hw, self.depth_fill_value, dtype=np.float32)
        results["depth_map"] = null_depth_map
        results["depth_map_path"] = ""
        results["depth_map_shape"] = img_shape_hw
        results["ori_depth_shape"] = img_shape_hw
        if "depth_fields" not in results:
            results["depth_fields"] = []
        if "depth_map" not in results["depth_fields"]:
            results["depth_fields"].append("depth_map")
        return results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(depth_fill_value={self.depth_fill_value})"


@TRANSFORMS.register_module()
class Albu(BaseTransform):
    """Albumentation augmentation."""

    def __init__(
        self,
        transforms: List[dict],
        bbox_params: Optional[dict] = None,
        keymap: Optional[dict] = None,
        skip_img_without_anno: bool = False,
    ) -> None:
        if Compose is None:
            raise RuntimeError("albumentations is not installed")

        transforms = copy.deepcopy(transforms)
        if bbox_params is not None:
            bbox_params = copy.deepcopy(bbox_params)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.skip_img_without_anno = skip_img_without_anno
        self.original_bboxes_were_rotated = False

        if isinstance(bbox_params, dict) and "label_fields" in bbox_params and "filter_lost_elements" in bbox_params:
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params["label_fields"][:]
            bbox_params["label_fields"] = ["idx_mapper"]
            del bbox_params["filter_lost_elements"]

        self.bbox_params = self.albu_builder(bbox_params) if bbox_params else None
        self.aug = Compose(
            [self.albu_builder(t) for t in self.transforms],
            bbox_params=self.bbox_params,
        )

        if not keymap:
            self.keymap_to_albu = {
                "img": "image",
                "gt_masks": "masks",
                "gt_bboxes": "bboxes",
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations."""
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()
        obj_type = args.pop("type")
        if is_str(obj_type):
            if albumentations is None:
                raise RuntimeError("albumentations is not installed")
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

        if "transforms" in args:
            args["transforms"] = [self.albu_builder(transform) for transform in args["transforms"]]

        return obj_cls(**args)

    @staticmethod
    def mapper(d: dict, keymap: dict) -> dict:
        """Dictionary mapper. Renames keys according to keymap provided."""
        updated_dict = {}
        for k, v in d.items():
            new_k = keymap.get(k, k)
            updated_dict[new_k] = v
        return updated_dict

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function of Albu."""
        results_albu = self.mapper(results, self.keymap_to_albu)
        results_albu, _ = self._preprocess_results(results_albu)

        if results_albu is None:
            if self.skip_img_without_anno:
                return None

        self.albu_results_backup = {}
        if self.filter_lost_elements:
            for field in self.origin_label_fields:
                if field in results_albu:
                    self.albu_results_backup[field] = copy.deepcopy(results_albu[field])

        albu_data = self.aug(**results_albu)
        results_albu = self._postprocess_results(albu_data, None)

        if results_albu is None:
            return None

        results_final = self.mapper(results_albu, self.keymap_back)
        for k, v in results.items():
            if k not in results_final:
                results_final[k] = v

        if "img" in results_final:
            results_final["img_shape"] = results_final["img"].shape[:2]

        return results_final

    def _preprocess_results(self, results: dict) -> tuple:
        """Pre-processing results to facilitate the use of Albu."""
        if "bboxes" in results:
            if not isinstance(results["bboxes"], BaseBoxes):
                raise TypeError(f"results['bboxes'] must be BaseBoxes, got {type(results['bboxes'])}")

            self.original_bboxes_were_rotated = isinstance(results["bboxes"], RotatedBoxes)
            if self.original_bboxes_were_rotated:
                # Convert RotatedBoxes to HorizontalBoxes for albumentations
                hbboxes_tensor = results["bboxes"].convert_to("hbox").tensor
            elif isinstance(results["bboxes"], HorizontalBoxes):
                hbboxes_tensor = results["bboxes"].tensor
            else:
                print_log(
                    f"Warning: Unexpected box type {type(results['bboxes'])} in Albu. "
                    "Attempting to convert to HorizontalBoxes.",
                    logger="current",
                    level=logging.WARNING,
                )
                try:
                    hbboxes_tensor = results["bboxes"].convert_to("hbox").tensor
                except Exception as e:
                    raise TypeError(f"Could not convert {type(results['bboxes'])} to HorizontalBoxes: {e}")

            results["bboxes"] = hbboxes_tensor.numpy().tolist()

            if self.filter_lost_elements:
                results["idx_mapper"] = np.arange(len(results["bboxes"])).tolist()

        if "masks" in results:
            if isinstance(results["masks"], PolygonMasks):
                print_log(
                    "Albu only supports Bitmap masks now. PolygonMasks will be ignored.",
                    logger="current",
                    level=logging.WARNING,
                )
                results.pop("masks")
            elif isinstance(results["masks"], BitmapMasks):
                results["masks"] = [mask for mask in results["masks"].masks]
            else:
                print_log(
                    f"Unsupported mask type {type(results['masks'])} for Albu.",
                    logger="current",
                    level=logging.WARNING,
                )
                results.pop("masks")

        return results, None

    def _postprocess_results(
        self,
        results: dict,  # This is the dict returned by self.aug(**results_albu)
        ori_masks,
    ) -> Optional[dict]:
        """Post-processing Albu output."""
        if "bboxes" in results:
            albu_output_bboxes_list = results["bboxes"]
            if isinstance(albu_output_bboxes_list, tuple):
                albu_output_bboxes_list = list(albu_output_bboxes_list)

            if albu_output_bboxes_list:
                albu_output_bboxes_np = np.array(albu_output_bboxes_list, dtype=np.float32)
            else:
                albu_output_bboxes_np = np.empty((0, 4), dtype=np.float32)

            albu_output_bboxes_np = albu_output_bboxes_np.reshape(-1, 4)
            final_bboxes_np = albu_output_bboxes_np
            final_idx_mapper = results.get("idx_mapper")

            if self.filter_lost_elements:
                if final_idx_mapper is None:
                    print_log(
                        "Warning: 'idx_mapper' not found in albumentations output "
                        "while filter_lost_elements is True. Bboxes might not be filtered correctly.",
                        logger="current",
                        level=logging.WARNING,
                    )

                for label_field_name in self.origin_label_fields:
                    original_field_data = self.albu_results_backup.get(label_field_name)
                    if original_field_data is not None and final_idx_mapper is not None:
                        valid_original_indices = [idx for idx in final_idx_mapper if idx < len(original_field_data)]
                        results[label_field_name] = np.array([original_field_data[i] for i in valid_original_indices])
                    elif label_field_name in results:
                        if final_idx_mapper is not None and results[label_field_name] is not None:
                            try:
                                results[label_field_name] = np.array(
                                    [results[label_field_name][i] for i in final_idx_mapper]
                                )
                            except IndexError:
                                print_log(
                                    f"IndexError when filtering {label_field_name} using albu's idx_mapper. "
                                    "Original field might be shorter than expected or idx_mapper is stale.",
                                    logger="current",
                                    level=logging.WARNING,
                                )

            results["bboxes"] = HorizontalBoxes(final_bboxes_np)

            if self.original_bboxes_were_rotated:
                print_log(
                    "Input 'gt_bboxes' were RotatedBoxes but output from Albu "
                    "transform will be HorizontalBoxes. Rotation information "
                    "is lost or not updated by this Albu transform.",
                    logger="current",
                    level=logging.WARNING,
                )

            if self.skip_img_without_anno and not len(results["bboxes"]):
                return None
        elif self.skip_img_without_anno and "bboxes" in self.keymap_to_albu.values():
            return None

        if "masks" in results:
            final_masks_np_list = results["masks"]
            if self.filter_lost_elements and final_idx_mapper is not None:
                pass

            if final_masks_np_list:
                h, w = final_masks_np_list[0].shape[:2]
                results["masks"] = BitmapMasks(final_masks_np_list, h, w)
            else:
                h, w = results["image"].shape[:2]
                results["masks"] = BitmapMasks([], h, w)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + "("
        repr_str += f"transforms={self.transforms}, "
        repr_str += f"bbox_params={self.bbox_params}, "
        repr_str += f"keymap={self.keymap_to_albu}, "
        repr_str += f"skip_img_without_anno={self.skip_img_without_anno})"
        return repr_str


@TRANSFORMS.register_module()
class AlbuRGBD(BaseTransform):
    """Albumentation augmentation for RGB-D data.

    Applies transformations to RGB image only. Depth map and annotations
    are not transformed by albumentations.
    """

    def __init__(self, transforms: List[dict], keymap: Optional[dict] = None) -> None:
        if Compose is None:
            raise RuntimeError("albumentations is not installed")

        transforms = copy.deepcopy(transforms)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)

        self.transforms = transforms
        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {"img": "image"}
        else:
            self.keymap_to_albu = keymap
            if "img" not in self.keymap_to_albu and "image" not in self.keymap_to_albu.values():
                print_log(
                    "Warning: 'keymap' for AlbuRGBD does not contain a mapping for 'img' to 'image'. "
                    "Albumentations might not process the RGB image correctly.",
                    logger="current",
                    level=logging.WARNING,
                )

        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

        print_log(
            "AlbuRGBD initialized. Albumentations will only be applied to the RGB image ('img'). "
            "Depth map and annotations (bboxes, masks) will NOT be transformed by albumentations. "
            "Ensure 'transforms' list contains only pixel-level/color-space augmentations if "
            "geometric consistency with depth/annotations is required.",
            logger="current",
            level=logging.INFO,
        )

    def albu_builder(self, cfg: dict) -> albumentations:
        """Builds an albumentation transform from a config dict."""
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()
        obj_type = args.pop("type")
        if is_str(obj_type):
            if albumentations is None:
                raise RuntimeError("albumentations is not installed")
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f"type must be a str or valid type, but got {type(obj_type)}")

        if "transforms" in args:
            args["transforms"] = [self.albu_builder(transform) for transform in args["transforms"]]
        return obj_cls(**args)

    @staticmethod
    def mapper(d: dict, keymap: dict) -> dict:
        """Dictionary mapper. Renames keys according to keymap provided."""
        updated_dict = {}
        for k, v in d.items():
            new_k = keymap.get(k, k)
            updated_dict[new_k] = v
        return updated_dict

    def transform(self, results: dict) -> dict:
        """Transform function of AlbuRGBD."""
        if "img" not in results:
            print_log(
                "Warning: 'img' key not found in results. AlbuRGBD will do nothing.",
                logger="current",
                level=logging.WARNING,
            )
            return results

        albu_input_key = self.keymap_to_albu.get("img", "image")
        albu_input_data = {albu_input_key: results["img"]}

        try:
            augmented_data = self.aug(**albu_input_data)
        except Exception as e:
            print_log(
                f"Error during albumentations processing in AlbuRGBD: {e}",
                logger="current",
                level=logging.ERROR,
            )
            return results

        original_img_key = self.keymap_back.get(albu_input_key, "img")
        if albu_input_key in augmented_data:
            results[original_img_key] = augmented_data[albu_input_key]
            results["img_shape"] = results[original_img_key].shape[:2]
        else:
            print_log(
                f"Warning: Expected key '{albu_input_key}' not found in albumentations output. "
                "RGB image may not have been augmented.",
                logger="current",
                level=logging.WARNING,
            )

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + "("
        repr_str += f"transforms={self.transforms}, "
        repr_str += f"keymap={self.keymap_to_albu})"
        return repr_str


@TRANSFORMS.register_module()
class RandomResizeRGBD(BaseTransform):
    """Resize an image, depth map, and OBBs to a randomly chosen scale.

    This transform resizes the input image and its corresponding depth map
    to a randomly selected scale from a given range.
    The interpolation method is linear for the image and nearest for the depth map.
    The `results` dictionary is updated with the new scale factor.

    Required Keys:
        - img (np.ndarray): The image to be resized.
        - depth_map (np.ndarray): The depth map to be resized.
        - gt_bboxes (RotatedBoxes): The oriented bounding boxes.
        - img_shape (tuple)
        - ori_shape (tuple)

    Modified Keys:
        - img
        - depth_map
        - gt_bboxes
        - img_shape
        - scale
        - scale_factor
        - keep_ratio

    Args:
        scale (tuple[float, float]): The range of scaling factors. A scale will
            be randomly chosen from this range.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to True.
        clip_object_border (bool): Whether to clip the objects outside the
            border of the image. Defaults to True.
        interpolation (str): Interpolation method for image. Defaults to 'linear'.
        depth_interpolation (str): Interpolation method for depth map.
            Defaults to 'nearest'.
    """

    def __init__(
        self,
        scale: Tuple[float, float],
        prob: float = 1.0,
        keep_ratio: bool = True,
        clip_object_border: bool = True,
        interpolation: str = "bilinear",
        depth_interpolation: str = "nearest",
    ) -> None:
        from mmengine.utils import is_tuple_of

        assert is_tuple_of(scale, (float, int)) and len(scale) == 2, "scale must be a tuple of two floats."
        assert 0 < scale[0] <= scale[1], "scale bounds must be positive and in ascending order."

        self.scale = scale
        self.prob = prob
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        self.interpolation = interpolation
        self.depth_interpolation = depth_interpolation

    def _resize_img_and_depth(self, results: dict, scale_factor: float) -> None:
        """Resize image and depth map with a given scale factor."""
        # mmcv.imrescale handles scale factors correctly
        if "img" in results:
            img, scale_used = mmcv.imrescale(
                results["img"],
                scale_factor,
                return_scale=True,
                interpolation=self.interpolation,
            )
            results["img"] = img
            results["img_shape"] = img.shape[:2]
            # scale_used is the actual scale factor applied
            results["scale_factor"] = (scale_used, scale_used, scale_used, scale_used)

        if "depth_map" in results:
            results["depth_map"] = mmcv.imrescale(
                results["depth_map"],
                scale_factor,
                interpolation=self.depth_interpolation,
            )

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with `results['scale_factor']`."""
        if "gt_bboxes" in results:
            results["gt_bboxes"].rescale_(results["scale_factor"])
            if self.clip_object_border:
                results["gt_bboxes"].clip_(results["img_shape"])

    def transform(self, results: dict) -> dict:
        """Apply the random resize transformation."""
        if random.random() > self.prob:
            return results

        # Generate a random scale factor from the given range
        scale_factor = random.uniform(self.scale[0], self.scale[1])

        # Store the chosen scale and ratio info in results, similar to mmcv's Resize
        results["scale"] = scale_factor
        results["keep_ratio"] = self.keep_ratio

        # Perform resizing
        self._resize_img_and_depth(results, scale_factor)
        self._resize_bboxes(results)

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"scale={self.scale}, prob={self.prob}, keep_ratio={self.keep_ratio}, "
            f"clip_object_border={self.clip_object_border}, "
            f"interpolation={self.interpolation}, "
            f"depth_interpolation={self.depth_interpolation})"
        )


@TRANSFORMS.register_module()
class RandomCropRGBD(BaseTransform):
    """Randomly crop the image, depth map, and OBBs.

    Args:
        crop_size (tuple[int, int]): The desired crop size in (h, w) format.
        allow_negative_crop (bool): Whether to allow a crop that does not
            contain any part of a bounding box. Defaults to False.
    """

    def __init__(
        self,
        crop_size: Tuple[int, int],
        prob: float = 1.0,
        allow_negative_crop: bool = False,
    ) -> None:
        assert isinstance(crop_size, tuple) and len(crop_size) == 2, "crop_size must be a tuple of (h, w)."
        assert all(isinstance(i, int) and i > 0 for i in crop_size), "crop_size must contain positive integers."

        self.crop_size = crop_size
        self.prob = prob
        self.allow_negative_crop = allow_negative_crop

    def _get_crop_offset(self, ori_h: int, ori_w: int) -> Tuple[int, int]:
        """Calculate a random top-left corner for the crop."""
        crop_h, crop_w = self.crop_size
        margin_h = max(ori_h - crop_h, 0)
        margin_w = max(ori_w - crop_w, 0)
        offset_h = random.randint(0, margin_h)
        offset_w = random.randint(0, margin_w)
        return offset_h, offset_w

    def _crop_data(self, results: dict, offset: Tuple[int, int]) -> dict:
        """Crop image, depth map, and annotations."""
        offset_h, offset_w = offset
        crop_h, crop_w = self.crop_size

        # Crop image and depth map
        for key in ["img", "depth_map"]:
            if key in results:
                data_field = results[key]
                results[key] = data_field[offset_h : offset_h + crop_h, offset_w : offset_w + crop_w, ...]

        # Update shape information
        results["img_shape"] = results["img"].shape[:2]

        # Crop annotations (OBBs)
        if "gt_bboxes" in results and len(results["gt_bboxes"]) > 0:
            gt_bboxes = results["gt_bboxes"].clone()
            # Shift the coordinate system origin to the top-left of the crop
            gt_bboxes.translate_((-offset_w, -offset_h))

            # Clip the OBBs to the new image boundaries
            gt_bboxes.clip_(results["img_shape"])

            # Filter out OBBs that are too small or completely outside after clipping
            valid_inds = (gt_bboxes.tensor[:, 2] > 1) & (gt_bboxes.tensor[:, 3] > 1)
            results["gt_bboxes"] = gt_bboxes[valid_inds]
            if "gt_bboxes_labels" in results:
                if isinstance(results["gt_bboxes_labels"], torch.Tensor):
                    results["gt_bboxes_labels"] = results["gt_bboxes_labels"][valid_inds]
                else:
                    # Convert to tensor first, then apply indexing, then back to numpy if needed
                    labels_tensor = torch.tensor(results["gt_bboxes_labels"])
                    results["gt_bboxes_labels"] = labels_tensor[valid_inds].numpy()

        return results

    def transform(self, results: dict) -> dict:
        """Apply the random crop transformation."""
        ori_h, ori_w = results["img_shape"]
        crop_h, crop_w = self.crop_size

        if crop_h > ori_h or crop_w > ori_w:
            raise ValueError(f"Crop size {self.crop_size} is larger than image size {(ori_h, ori_w)}.")

        if random.random() > self.prob:
            return results

        # Try to find a crop that contains at least one OBB, if not allowing negative crops
        if not self.allow_negative_crop and "gt_bboxes" in results and len(results.get("gt_bboxes", [])) > 0:
            for _ in range(50):  # 50 attempts to find a valid crop
                offset_h, offset_w = self._get_crop_offset(ori_h, ori_w)
                crop_bbox_xyxy = [
                    offset_w,
                    offset_h,
                    offset_w + crop_w,
                    offset_h + crop_h,
                ]

                # Check for overlap between crop area and OBBs
                hbboxes = results["gt_bboxes"].convert_to("hbox").tensor
                from mmdet.structures.bbox import bbox_overlaps

                ious = bbox_overlaps(
                    hbboxes,
                    torch.tensor(crop_bbox_xyxy, device=hbboxes.device).unsqueeze(0),
                )

                if ious.max() > 0:
                    break
            else:  # Fallback to a completely random crop if no overlap found
                offset_h, offset_w = self._get_crop_offset(ori_h, ori_w)
        else:
            offset_h, offset_w = self._get_crop_offset(ori_h, ori_w)

        return self._crop_data(results, (offset_h, offset_w))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"crop_size={self.crop_size}, "
            f"prob={self.prob}, "
            f"allow_negative_crop={self.allow_negative_crop})"
        )


@TRANSFORMS.register_module()
class CoarseDropoutDepth(BaseTransform):
    """Apply CoarseDropout to the depth map only.

    This transform simulates sensor dropouts or occlusions in the depth channel
    by creating rectangular holes. The RGB image remains unchanged.

    Required Keys:
        - depth_map (np.ndarray)

    Modified Keys:
        - depth_map

    Args:
        max_holes (int): Maximum number of holes to draw. Defaults to 8.
        max_height (int): Maximum height of a hole. Defaults to 8.
        max_width (int): Maximum width of a hole. Defaults to 8.
        min_holes (int, optional): Minimum number of holes. Defaults to `max_holes`.
        min_height (int, optional): Minimum height of a hole. Defaults to `max_height`.
        min_width (int, optional): Minimum width of a hole. Defaults to `max_width`.
        fill_value (int or float): Value for the dropped pixels (holes). This
            should typically be the value representing invalid depth (e.g., 0).
            Defaults to 0.
    """

    def __init__(
        self,
        max_holes: int = 8,
        max_height: int = 8,
        max_width: int = 8,
        min_holes: Optional[int] = None,
        min_height: Optional[int] = None,
        min_width: Optional[int] = None,
        fill_value: Union[int, float] = 0,
        prob: float = 0.5,
    ):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.prob = prob

        assert 0 < self.min_holes <= self.max_holes
        assert 0 < self.min_height <= self.max_height
        assert 0 < self.min_width <= self.max_width
        assert 0.0 <= self.prob <= 1.0

    def transform(self, results: dict) -> dict:
        """Apply the CoarseDropout augmentation to the depth map."""
        if random.random() > self.prob:
            return results

        if "depth_map" not in results or results["depth_map"] is None:
            return results

        h, w = results["depth_map"].shape[:2]
        depth_map = results["depth_map"].copy()

        num_holes = random.randint(self.min_holes, self.max_holes)

        for _ in range(num_holes):
            hole_h = random.randint(self.min_height, self.max_height)
            hole_w = random.randint(self.min_width, self.max_width)

            if h - hole_h > 0:
                y1 = random.randint(0, h - hole_h)
            else:
                y1 = 0
            if w - hole_w > 0:
                x1 = random.randint(0, w - hole_w)
            else:
                x1 = 0
            y2 = y1 + hole_h
            x2 = x1 + hole_w

            depth_map[y1:y2, x1:x2] = self.fill_value

        results["depth_map"] = depth_map
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_holes={self.max_holes}, max_height={self.max_height}, "
            f"max_width={self.max_width}, min_holes={self.min_holes}, "
            f"min_height={self.min_height}, min_width={self.min_width}, "
            f"fill_value={self.fill_value}, prob={self.prob})"
        )


@TRANSFORMS.register_module()
class RandomSizedCropRGBD(BaseTransform):
    """Randomly crop a sub-image of a certain scale, then resize it back.

    This transform first samples a random scale from the given range. It then
    performs a random crop of the input image, depth map, and annotations
    at that scale, maintaining the original aspect ratio. Finally, it resizes
    the cropped region back to the original image size.

    If the cropped area contains no objects, it will be resampled up to
    `max_resample_attempts` times.

    This is useful for learning representations that are robust to objects
    appearing at different scales.

    Required Keys in results:
        - img (np.ndarray): The RGB image.
        - depth_map (np.ndarray, optional): The depth map.
        - gt_bboxes (RotatedBoxes): The oriented bounding boxes.
        - img_shape (tuple): The shape of the image.

    Modified Keys in results:
        - img
        - depth_map
        - gt_bboxes
        - gt_bboxes_labels
        - img_shape

    Args:
        min_max_height (tuple[int, int]): The range of heights to sample for
            the crop, in pixels. The width will be scaled proportionally to
            maintain the aspect ratio.
        height (int): The final height to resize the cropped image to. This
            should match the model's expected input height.
        width (int): The final width to resize the cropped image to. This
            should match the model's expected input width.
        prob (float): The probability of performing this transform.
            Defaults to 1.0.
        max_resample_attempts (int): The maximum number of times to resample
            the crop if it contains no objects. Defaults to 10.
        interpolation (str): Interpolation method for resizing the RGB image.
            Defaults to 'bilinear'.
        depth_interpolation (str): Interpolation method for resizing the
            depth map. Defaults to 'nearest'.
    """

    def __init__(
        self,
        min_max_height: Tuple[int, int],
        height: int,
        width: int,
        prob: float = 1.0,
        max_resample_attempts: int = 10,
        interpolation: str = "bilinear",
        depth_interpolation: str = "nearest",
    ) -> None:
        assert isinstance(min_max_height, tuple) and len(min_max_height) == 2
        assert 0 < min_max_height[0] <= min_max_height[1]
        assert height > 0 and width > 0
        assert 0.0 <= prob <= 1.0
        assert max_resample_attempts >= 0

        self.min_max_height = min_max_height
        self.height = height
        self.width = width
        self.prob = prob
        self.max_resample_attempts = max_resample_attempts
        self.interpolation = interpolation
        self.depth_interpolation = depth_interpolation

    def _get_random_crop_params(self, ori_h: int, ori_w: int) -> Tuple[int, int, int, int]:
        """Determine random crop height, width, and offset."""
        # 1. Determine crop size based on random height
        crop_h = random.randint(self.min_max_height[0], self.min_max_height[1])
        aspect_ratio = ori_w / ori_h
        crop_w = int(round(crop_h * aspect_ratio))

        # Ensure crop size is not larger than original size
        crop_h = min(crop_h, ori_h)
        crop_w = min(crop_w, ori_w)

        # 2. Determine crop offset
        margin_h = max(ori_h - crop_h, 0)
        margin_w = max(ori_w - crop_w, 0)
        offset_h = random.randint(0, margin_h)
        offset_w = random.randint(0, margin_w)

        return crop_h, crop_w, offset_h, offset_w

    def _transform_annotations(self, results: dict, crop_h: int, crop_w: int, offset_h: int, offset_w: int) -> dict:
        """Transforms annotations for a given crop."""
        if "gt_bboxes" in results and len(results.get("gt_bboxes", [])) > 0:
            gt_bboxes = results["gt_bboxes"].clone()

            gt_bboxes.translate_((-offset_w, -offset_h))

            w_scale_factor = self.width / crop_w
            h_scale_factor = self.height / crop_h
            scale_factor_xyxy = (w_scale_factor, h_scale_factor, w_scale_factor, h_scale_factor)
            gt_bboxes.rescale_(scale_factor_xyxy)

            gt_bboxes.clip_((self.height, self.width))

            valid_inds = (gt_bboxes.tensor[:, 2] > 1) & (gt_bboxes.tensor[:, 3] > 1)
            results["gt_bboxes"] = gt_bboxes[valid_inds]
            if "gt_bboxes_labels" in results:
                labels = results["gt_bboxes_labels"]
                if isinstance(labels, torch.Tensor):
                    results["gt_bboxes_labels"] = labels[valid_inds]
                else:
                    results["gt_bboxes_labels"] = np.array(labels)[valid_inds.cpu().numpy()]
        return results

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        ori_h, ori_w = results["img_shape"]
        original_bboxes = results.get("gt_bboxes")

        for i in range(self.max_resample_attempts + 1):
            crop_h, crop_w, offset_h, offset_w = self._get_random_crop_params(ori_h, ori_w)

            # --- アノテーションの有効性を先にチェック ---
            # もしアノテーションがなければ、どんなクロップでも有効とみなす
            if original_bboxes is None or len(original_bboxes) == 0:
                break

            # クロップ後のアノテーションを計算（画像を実際にクロップする前に）
            temp_results = copy.deepcopy(results)  # ここでcopy.deepcopyを使用
            temp_results = self._transform_annotations(temp_results, crop_h, crop_w, offset_h, offset_w)

            # クロップ後に有効なBBoxが1つ以上残っているか確認
            if len(temp_results.get("gt_bboxes", [])) > 0:
                # 有効なクロップが見つかったのでループを抜ける
                break

            # 最後の試行でも見つからなかった場合
            if i == self.max_resample_attempts:
                # 諦めて最後のクロップパラメータを使用する
                # この場合、結果としてアノテーションが空になる
                break

        # --- 決定したパラメータで画像と深度マップを実際に変換 ---
        img = results["img"][offset_h : offset_h + crop_h, offset_w : offset_w + crop_w]
        if "depth_map" in results and results.get("depth_map") is not None:
            depth_map = results["depth_map"][offset_h : offset_h + crop_h, offset_w : offset_w + crop_w]
        else:
            depth_map = None

        target_size = (self.width, self.height)
        img_resized = mmcv.imresize(img, target_size, interpolation=self.interpolation, backend="cv2")
        if depth_map is not None:
            depth_map_resized = mmcv.imresize(
                depth_map, target_size, interpolation=self.depth_interpolation, backend="cv2"
            )
        else:
            depth_map_resized = None

        results["img"] = img_resized
        results["depth_map"] = depth_map_resized
        results["img_shape"] = (self.height, self.width)

        # --- 最終的なアノテーションをresultsに格納 ---
        results = self._transform_annotations(results, crop_h, crop_w, offset_h, offset_w)

        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"min_max_height={self.min_max_height}, "
            f"height={self.height}, width={self.width}, "
            f"prob={self.prob}, "
            f"max_resample_attempts={self.max_resample_attempts}, "
            f"interpolation='{self.interpolation}', "
            f"depth_interpolation='{self.depth_interpolation}')"
        )


@TRANSFORMS.register_module()
class RandomDepthOffset(BaseTransform):
    """Add a random offset to the entire depth map.

    This transform simulates a global change in depth perception, for example,
    due to sensor calibration drift or changes in lighting conditions that
    affect the entire scene's depth reading.

    Required Keys:
        - depth_map (np.ndarray)

    Modified Keys:
        - depth_map

    Args:
        offset_range (tuple[float, float]): The range from which to sample the
            random offset value (in the same unit as the depth map, e.g., meters).
            The offset will be sampled uniformly from [min_offset, max_offset].
            For example, (-0.1, 0.1) means an offset between -10cm and +10cm
            will be added.
        prob (float): The probability of applying this transform. Defaults to 0.5.
        invalid_depth_value (float): The value in the depth map that represents
            invalid or no-return pixels (e.g., 0.0). These pixels will not
            be modified by the offset. Defaults to 0.0.
        clamp_to_zero (bool): Whether to clamp negative depth values to zero
            after applying the offset. Defaults to True.
    """

    def __init__(
        self,
        offset_range: Tuple[float, float],
        prob: float = 0.5,
        invalid_depth_value: float = 0.0,
        clamp_to_zero: bool = True,
    ):
        assert isinstance(offset_range, tuple) and len(offset_range) == 2, \
            "offset_range must be a tuple of two floats."
        assert offset_range[0] <= offset_range[1], \
            "The first value in offset_range must be less than or equal to the second."
        assert 0.0 <= prob <= 1.0, "prob must be between 0.0 and 1.0."

        self.offset_range = offset_range
        self.prob = prob
        self.invalid_depth_value = invalid_depth_value
        self.clamp_to_zero = clamp_to_zero

    def transform(self, results: dict) -> dict:
        """Apply the random depth offset augmentation."""
        if random.random() > self.prob:
            return results

        if "depth_map" not in results or results["depth_map"] is None:
            return results

        depth_map = results["depth_map"].copy()

        # Generate a random offset from the specified range
        random_offset = random.uniform(self.offset_range[0], self.offset_range[1])

        # Create a mask for valid depth pixels (i.e., not the invalid_depth_value)
        valid_mask = (depth_map != self.invalid_depth_value)

        # Add the offset only to the valid pixels
        depth_map[valid_mask] += random_offset

        # Ensure that the depth values do not become negative after adding the offset.
        if self.clamp_to_zero:
            depth_map[depth_map < 0] = 0

        results["depth_map"] = depth_map
        return results

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"offset_range={self.offset_range}, "
            f"prob={self.prob}, "
            f"invalid_depth_value={self.invalid_depth_value}, "
            f"clamp_to_zero={self.clamp_to_zero})"
        )
