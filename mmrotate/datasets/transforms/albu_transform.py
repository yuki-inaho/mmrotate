# my_project/datasets/pipelines/albu_rotate.py

try:
    import albumentations
except ImportError:
    albumentations = None
import numpy as np
import torch

from mmrotate.registry import TRANSFORMS
from mmrotate.structures import QuadriBoxes
from mmdet.datasets.transforms import Albu


@TRANSFORMS.register_module()
class AlbuRotate(Albu):
    """
    Albumentations wrapper for rotated object detection.

    This wrapper converts quadrilateral bounding boxes (qbox, 8-dim)
    to keypoints, applies albumentations transforms, and then converts
    the transformed keypoints back to quadrilateral bounding boxes.
    """

    def __init__(self, transforms, bbox_params=None, keymap=None, skip_img_without_anno=False):
        # In this wrapper, we manually handle bboxes as keypoints, so we ensure bbox_params is None.
        if bbox_params is not None:
            raise ValueError("bbox_params is not supported in AlbuRotate, use keypoint_params for qbox.")

        # Default keymap for AlbuRotate
        if keymap is None:
            keymap = {
                "img": "image",
                "gt_bboxes": "keypoints",  # qbox will be converted to keypoints
                "gt_labels": "keypoint_labels",  # labels for keypoints
            }

        super().__init__(
            transforms=transforms,
            bbox_params=None,  # Manually handled via keypoints
            keymap=keymap,
            skip_img_without_anno=skip_img_without_anno,
        )

        # AlbuRotate specific keypoint parameters
        self.keypoint_params = {"type": "KeypointParams", "format": "xy", "remove_invisible": False}

    def transform(self, results):
        """Transform method override for rotated bounding boxes."""
        # Pre-process: Convert qbox to keypoints for albumentations
        img = results["img"]
        gt_bboxes = results.get("gt_bboxes", None)  # QuadriBoxes or qbox format (N, 8)
        gt_labels = results.get("gt_labels", None)

        # Handle both QuadriBoxes and numpy array input
        if gt_bboxes is not None and hasattr(gt_bboxes, "tensor"):
            # QuadriBoxes from LoadAnnotations
            qbox_array = gt_bboxes.tensor.cpu().numpy()
            is_base_boxes = True
        elif gt_bboxes is not None:
            # Already numpy array
            qbox_array = gt_bboxes
            is_base_boxes = False
        else:
            qbox_array = None
            is_base_boxes = False

        if qbox_array is None or len(qbox_array) == 0:
            # If no ground truth, just transform the image
            if self.skip_img_without_anno:
                return results
            # Apply only image transformation
            augmented = self.aug(image=img)
            results["img"] = augmented["image"]
            results["img_shape"] = results["img"].shape[:2]
            return results

        # Convert qboxes to keypoints format for albumentations
        # From (N, 8) to (N*4, 2) - reshape each qbox to 4 keypoints
        keypoints = qbox_array.reshape(-1, 2)

        # Albumentations requires a label for each keypoint.
        # We repeat each bbox label 4 times for its 4 vertices.
        keypoint_labels = np.repeat(gt_labels, 4)

        # Apply albumentations transforms with keypoint support
        try:
            # Create temporary albumentations composer with keypoint params
            keypoint_aug = albumentations.Compose(
                transforms=self.aug.transforms,
                keypoint_params=albumentations.KeypointParams(
                    format="xy", remove_invisible=False, label_fields=["keypoint_labels"]
                ),
            )

            augmented = keypoint_aug(image=img, keypoints=keypoints.tolist(), keypoint_labels=keypoint_labels.tolist())
        except Exception as e:
            # Fallback: if keypoint transform fails, only transform image
            print(f"Warning: Keypoint transform failed ({e}), applying image-only transform")
            augmented = self.aug(image=img)
            augmented["keypoints"] = keypoints.tolist()
            augmented["keypoint_labels"] = keypoint_labels.tolist()

        # Post-process: Convert keypoints back to qbox
        results["img"] = augmented["image"]
        results["img_shape"] = results["img"].shape[:2]

        # Reconstruct qboxes from transformed keypoints
        aug_keypoints = np.array(augmented["keypoints"], dtype=np.float32)

        # Handle cases where keypoint transformation fails and labels are None
        if augmented.get("keypoint_labels") is None or len(augmented.get("keypoint_labels", [])) == 0:
            aug_labels = np.array([], dtype=np.int64)
        else:
            aug_labels = np.array(augmented["keypoint_labels"], dtype=np.int64)

        if aug_keypoints.shape[0] == 0 or aug_keypoints.shape[0] % 4 != 0:
            # All bboxes were removed or invalid keypoint count
            if is_base_boxes:
                # Return empty QuadriBoxes to maintain type consistency
                results["gt_bboxes"] = QuadriBoxes(torch.zeros((0, 8), dtype=torch.float32))
            else:
                results["gt_bboxes"] = np.zeros((0, 8), dtype=np.float32)
            results["gt_labels"] = np.array([], dtype=np.int64)
        else:
            # Reshape back to (M, 8) qbox format
            transformed_qboxes = aug_keypoints.reshape(-1, 8)

            if is_base_boxes:
                # Convert back to QuadriBoxes to maintain type consistency for ConvertBoxType
                results["gt_bboxes"] = QuadriBoxes(torch.from_numpy(transformed_qboxes).float())
            else:
                results["gt_bboxes"] = transformed_qboxes

            # Update labels based on the remaining keypoints (every 4th label)
            results["gt_labels"] = aug_labels[::4]

        return results
