# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, TypeVar, Union, Sequence
import logging

import cv2
import numpy as np
import torch
from mmdet.structures.bbox import BaseBoxes, register_box
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torch import BoolTensor, Tensor
from mmengine.logging import print_log


T = TypeVar("T")
DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]


@register_box("rbox")
class RotatedBoxes(BaseBoxes):
    """The rotated box class used in MMRotate by default.

    The ``box_dim`` of ``RotatedBoxes`` is 5, which means the length of the
    last dimension of the input should be 5. Each row of data means
    (x, y, w, h, t), where 'x' and 'y' are the coordinates of the box center,
    'w' and 'h' are the length of box sides, 't' is the box angle represented
    in radian. A rotated box can be regarded as rotating the horizontal box
    (x, y, w, h) w.r.t its center by 't' radian CW.

    Args:
        data (Tensor or np.ndarray or Sequence): The box data with shape
            (..., 5).
        dtype (torch.dtype, Optional): data type of boxes. Defaults to None.
        device (str or torch.device, Optional): device of boxes.
            Default to None.
        clone (bool): Whether clone ``boxes`` or not. Defaults to True.
    """

    box_dim = 5

    def __init__(
        self,
        data: Union[Tensor, np.ndarray, Sequence],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        clone: bool = True,
    ) -> None:
        super().__init__(data, dtype=dtype, device=device, clone=clone)
        if self.tensor.ndim == 2 and self.tensor.size(-1) != 5:
            raise ValueError(
                f"The last dimension of rotated_boxes tensor should be 5, but got {self.tensor.size(-1)}"
            )

    def regularize_boxes(
        self,
        pattern: Optional[str] = None,
        width_longer: bool = True,
        start_angle: float = -90,
    ) -> Tensor:
        boxes = self.tensor
        if pattern is not None:
            if pattern == "oc":
                width_longer, start_angle = False, -90
            elif pattern == "le90":
                width_longer, start_angle = True, -90
            elif pattern == "le135":
                width_longer, start_angle = True, -45
            else:
                raise ValueError(
                    f"pattern only can be 'oc', 'le90', and'le135', but get {pattern}."
                )
        start_angle = start_angle / 180 * np.pi

        x, y, w, h, t = boxes.unbind(dim=-1)
        if width_longer:
            w_ = torch.where(w > h, w, h)
            h_ = torch.where(w > h, h, w)
            t = torch.where(w > h, t, t + np.pi / 2)
            t = ((t - start_angle) % np.pi) + start_angle
        else:
            t = (t - start_angle) % np.pi
            w_ = torch.where(t < np.pi / 2, w, h)
            h_ = torch.where(t < np.pi / 2, h, w)
            t = torch.where(t < np.pi / 2, t, t - np.pi / 2) + start_angle
        self.tensor = torch.stack([x, y, w_, h_, t], dim=-1)
        return self.tensor

    @property
    def centers(self) -> Tensor:
        return self.tensor[..., :2]

    @property
    def areas(self) -> Tensor:
        return self.tensor[..., 2] * self.tensor[..., 3]

    @property
    def widths(self) -> Tensor:
        return self.tensor[..., 2]

    @property
    def heights(self) -> Tensor:
        return self.tensor[..., 3]

    def flip_(self, img_shape: Tuple[int, int], direction: str = "horizontal") -> None:
        assert direction in ["horizontal", "vertical", "diagonal"]
        flipped = self.tensor
        if direction == "horizontal":
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 4] = -flipped[..., 4]
        elif direction == "vertical":
            flipped[..., 1] = img_shape[0] - flipped[..., 1]
            flipped[..., 4] = -flipped[..., 4]
        else:
            flipped[..., 0] = img_shape[1] - flipped[..., 0]
            flipped[..., 1] = img_shape[0] - flipped[..., 1]

    def translate_(self, distances: Tuple[float, float]) -> None:
        boxes = self.tensor
        assert len(distances) == 2
        boxes[..., :2] = boxes[..., :2] + boxes.new_tensor(distances)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        if self.tensor.numel() > 0:  # Avoid warning for empty tensors
            pass
        #    warnings.warn("The `clip` function does nothing in `RotatedBoxes`.")

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        boxes = self.tensor
        rotation_matrix = boxes.new_tensor(cv2.getRotationMatrix2D(center, -angle, 1))

        centers, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        t = t + angle / 180 * np.pi
        centers = torch.cat([centers, centers.new_ones(*centers.shape[:-1], 1)], dim=-1)
        centers_T = torch.transpose(centers, -1, -2)
        centers_T = torch.matmul(rotation_matrix, centers_T)
        centers = torch.transpose(centers_T, -1, -2)
        self.tensor = torch.cat([centers, wh, t], dim=-1)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        boxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = boxes.new_tensor(homography_matrix)
        corners = self.rbox2corner(boxes)
        corners = torch.cat([corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        corners = corners[..., :2] / corners[..., 2:3]
        self.tensor = self.corner2rbox(corners)

    @staticmethod
    def rbox2corner(boxes: Tensor) -> Tensor:
        ctr, w, h, theta = torch.split(boxes, (2, 1, 1, 1), dim=-1)
        cos_value, sin_value = torch.cos(theta), torch.sin(theta)
        vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
        vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return torch.stack([pt1, pt2, pt3, pt4], dim=-2)

    @staticmethod
    def corner2rbox(corners: Tensor) -> Tensor:
        original_shape = corners.shape[:-2]
        points = corners.cpu().numpy().reshape(-1, 4, 2)
        rboxes = []
        for pts in points:
            (x, y), (w, h), angle = cv2.minAreaRect(pts)
            rboxes.append([x, y, w, h, angle / 180 * np.pi])
        rboxes = corners.new_tensor(rboxes)
        return rboxes.reshape(*original_shape, 5)

    def rescale_(
        self,
        scale_factor: Union[float, Tuple[float, float], Sequence[float], np.ndarray],
    ) -> None:  # ★ np.ndarray を追加
        """Rescale boxes w.r.t. origin according to scale_factor.
        # ... (docstring は変更なし) ...
        """
        original_scale_factor_for_log = scale_factor
        processed_scale_factor = None

        if isinstance(scale_factor, (float, int)):
            processed_scale_factor = (float(scale_factor), float(scale_factor))
        elif isinstance(
            scale_factor, (list, tuple, np.ndarray)
        ):  # ★ np.ndarray をここに追加
            if len(scale_factor) == 4:
                print_log(
                    f"RotatedBoxes.rescale_: Received {len(scale_factor)}-element scale_factor {original_scale_factor_for_log}. "
                    f"Using first two elements: ({float(scale_factor[0])}, {float(scale_factor[1])})",
                    logger="current",
                    level=logging.DEBUG,
                )
                processed_scale_factor = (
                    float(scale_factor[0]),
                    float(scale_factor[1]),
                )
            elif len(scale_factor) == 2:
                processed_scale_factor = (
                    float(scale_factor[0]),
                    float(scale_factor[1]),
                )

        if processed_scale_factor is None:  # If none of the above conditions were met
            raise TypeError(
                f"scale_factor should be a number, or a tuple/list/ndarray of length 2 or 4, "
                f"got {original_scale_factor_for_log} of type {type(original_scale_factor_for_log)}"
            )

        if self.tensor.numel() == 0:
            return

        boxes = self.tensor
        scale_x, scale_y = (
            processed_scale_factor[0],
            processed_scale_factor[1],
        )  # ★ Use processed_scale_factor
        ctrs, w, h, t = torch.split(boxes, [2, 1, 1, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)

        ctrs = ctrs * ctrs.new_tensor([scale_x, scale_y])
        w = w * torch.sqrt((scale_x * cos_value) ** 2 + (scale_y * sin_value) ** 2)
        h = h * torch.sqrt((scale_x * sin_value) ** 2 + (scale_y * cos_value) ** 2)

        t_new = torch.atan2(sin_value * scale_y, cos_value * scale_x)
        self.tensor = torch.cat([ctrs, w, h, t_new], dim=-1)

    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        boxes = self.tensor
        assert len(scale_factor) == 2
        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        scale_factor = boxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        self.tensor = torch.cat([ctrs, wh, t], dim=-1)

    def is_inside(
        self,
        img_shape: Tuple[int, int],
        all_inside: bool = False,
        allowed_border: int = 0,
    ) -> BoolTensor:
        img_h, img_w = img_shape
        boxes = self.tensor
        return (
            (boxes[..., 0] <= img_w + allowed_border)
            & (boxes[..., 1] <= img_h + allowed_border)
            & (boxes[..., 0] >= -allowed_border)
            & (boxes[..., 1] >= -allowed_border)
        )

    def find_inside_points(
        self, points: Tensor, is_aligned: bool = False, eps: float = 0.01
    ) -> BoolTensor:
        boxes = self.tensor
        assert boxes.dim() == 2

        if not is_aligned:
            boxes = boxes[None, :, :]
            points = points[:, None, :]
        else:
            assert boxes.size(0) == points.size(0)

        ctrs, wh, t = torch.split(boxes, [2, 2, 1], dim=-1)
        cos_value, sin_value = torch.cos(t), torch.sin(t)
        matrix = torch.cat(
            [cos_value, sin_value, -sin_value, cos_value], dim=-1
        ).reshape(*boxes.shape[:-1], 2, 2)

        offset = points - ctrs
        offset = torch.matmul(matrix, offset[..., None])
        offset = offset.squeeze(-1)
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        w, h = wh[..., 0], wh[..., 1]
        return (
            (offset_x <= w / 2 - eps)
            & (offset_x >= -w / 2 + eps)
            & (offset_y <= h / 2 - eps)
            & (offset_y >= -h / 2 + eps)
        )

    @staticmethod
    def overlaps(
        boxes1: BaseBoxes,
        boxes2: BaseBoxes,
        mode: str = "iou",
        is_aligned: bool = False,
        eps: float = 1e-6,
    ) -> Tensor:
        from mmrotate.structures.bbox import rbbox_overlaps

        boxes1_converted = boxes1.convert_to("rbox")
        boxes2_converted = boxes2.convert_to("rbox")
        return rbbox_overlaps(
            boxes1_converted.tensor,
            boxes2_converted.tensor,
            mode=mode,
            is_aligned=is_aligned,
            eps=eps,
        )

    @staticmethod
    def from_instance_masks(masks: MaskType) -> "RotatedBoxes":
        num_masks = len(masks)
        if num_masks == 0:
            return RotatedBoxes([], dtype=torch.float32)

        boxes = []
        if isinstance(masks, BitmapMasks):
            for idx in range(num_masks):
                mask = masks.masks[idx]
                if mask.dtype != np.uint8:
                    mask = mask.astype(np.uint8)
                if not (
                    np.array_equal(np.unique(mask), np.array([0]))
                    or np.array_equal(np.unique(mask), np.array([1]))
                    or np.array_equal(np.unique(mask), np.array([0, 1]))
                ):
                    mask = (mask > 0).astype(np.uint8)

                points = np.stack(np.nonzero(mask), axis=-1).astype(np.float32)
                if points.shape[0] < 3:
                    ys, xs = np.nonzero(mask)
                    if len(xs) == 0:
                        boxes.append([0, 0, 0, 0, 0])
                        continue
                    x_min, x_max = np.min(xs), np.max(xs)
                    y_min, y_max = np.min(ys), np.max(ys)
                    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
                    w, h = (x_max - x_min) + 1, (y_max - y_min) + 1
                    boxes.append([cx, cy, w, h, 0.0])
                    continue
                (x, y), (w, h), angle_deg = cv2.minAreaRect(points)
                angle_rad = angle_deg * np.pi / 180.0
                boxes.append([x, y, w, h, angle_rad])
        elif isinstance(masks, PolygonMasks):
            for idx, poly_per_obj in enumerate(masks.masks):
                pts_per_obj_list = []
                for p in poly_per_obj:
                    pts_per_obj_list.append(
                        np.array(p, dtype=np.float32).reshape(-1, 2)
                    )
                if not pts_per_obj_list:
                    boxes.append([0, 0, 0, 0, 0])
                    continue
                pts_per_obj = np.concatenate(pts_per_obj_list, axis=0)
                if pts_per_obj.shape[0] < 3:
                    if len(pts_per_obj) == 0:
                        boxes.append([0, 0, 0, 0, 0])
                        continue
                    x_coords = pts_per_obj[:, 0]
                    y_coords = pts_per_obj[:, 1]
                    cx, cy = np.mean(x_coords), np.mean(y_coords)
                    w = (
                        np.max(x_coords) - np.min(x_coords) + 1
                        if len(x_coords) > 0
                        else 0
                    )
                    h = (
                        np.max(y_coords) - np.min(y_coords) + 1
                        if len(y_coords) > 0
                        else 0
                    )
                    boxes.append([cx, cy, w, h, 0.0])
                    continue
                (x, y), (w, h), angle_deg = cv2.minAreaRect(pts_per_obj)
                angle_rad = angle_deg * np.pi / 180.0
                boxes.append([x, y, w, h, angle_rad])
        else:
            raise TypeError(
                f"`masks` must be `BitmapMasks`  or `PolygonMasks`, but got {type(masks)}."
            )
        return RotatedBoxes(boxes, dtype=torch.float32)
