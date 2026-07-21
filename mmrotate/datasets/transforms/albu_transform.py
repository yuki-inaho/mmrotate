# Copyright (c) OpenMMLab. All rights reserved.
# my_project/datasets/pipelines/albu_rotate.py

try:
    import albumentations
except ImportError:
    albumentations = None
import numpy as np
import torch
from mmdet.datasets.transforms import Albu

from mmrotate.registry import TRANSFORMS
from mmrotate.structures import QuadriBoxes


@TRANSFORMS.register_module()
class AlbuRotate(Albu):
    """Albumentations wrapper for rotated object detection.

    This wrapper converts quadrilateral bounding boxes (qbox, 8-dim) to
    keypoints, applies albumentations transforms, and then converts the
    transformed keypoints back to quadrilateral bounding boxes.
    """

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 skip_img_without_anno=False):
        # This wrapper manually handles bboxes as keypoints.
        if bbox_params is not None:
            raise ValueError('bbox_params is not supported in AlbuRotate, use '
                             'keypoint_params for qbox.')

        # Default keymap for AlbuRotate
        if keymap is None:
            keymap = {
                'img': 'image',
                # qbox will be converted to keypoints.
                'gt_bboxes': 'keypoints',
                # Kept locally, not passed to albumentations.
                'gt_labels': 'keypoint_labels',
            }

        super().__init__(
            transforms=transforms,
            bbox_params=None,  # Manually handled via keypoints
            keymap=keymap,
            skip_img_without_anno=skip_img_without_anno,
        )

        # AlbuRotate specific keypoint parameters
        self.keypoint_params = {
            'type': 'KeypointParams',
            'format': 'xy',
            'remove_invisible': False
        }

    def transform(self, results):
        """Transform method override for rotated bounding boxes."""
        # Pre-process: Convert qbox to keypoints for albumentations
        img = results['img']
        # QuadriBoxes or qbox format (N, 8).
        gt_bboxes = results.get('gt_bboxes', None)
        gt_labels = results.get('gt_labels', None)

        # Handle both QuadriBoxes and numpy array input
        if gt_bboxes is not None and hasattr(gt_bboxes, 'tensor'):
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
            results['img'] = augmented['image']
            results['img_shape'] = results['img'].shape[:2]
            return results

        if gt_labels is None:
            label_array = np.zeros((len(qbox_array), ), dtype=np.int64)
        elif hasattr(gt_labels, 'cpu'):
            label_array = gt_labels.cpu().numpy().astype(np.int64)
        else:
            label_array = np.asarray(gt_labels, dtype=np.int64)

        # Convert qboxes to keypoints format for albumentations.
        # Labels are bbox-level metadata and should not be passed as keypoint
        # labels:
        # Albumentations 2.x validates one label per keypoint and can collapse
        # the repeated labels back to 4 entries, causing the transform to fail.
        keypoints = qbox_array.reshape(-1, 2)

        # Apply albumentations transforms with keypoint support
        try:
            # Create temporary albumentations composer with keypoint params
            keypoint_aug = albumentations.Compose(
                transforms=self.aug.transforms,
                keypoint_params=albumentations.KeypointParams(
                    format='xy', remove_invisible=False),
            )

            augmented = keypoint_aug(image=img, keypoints=keypoints.tolist())
        except Exception as e:
            # Fallback: if keypoint transform fails, only transform image
            print(f'Warning: Keypoint transform failed ({e}), applying '
                  'image-only transform')
            augmented = self.aug(image=img)
            augmented['keypoints'] = keypoints.tolist()

        # Post-process: Convert keypoints back to qbox
        results['img'] = augmented['image']
        results['img_shape'] = results['img'].shape[:2]

        # Reconstruct qboxes from transformed keypoints
        aug_keypoints = np.array(augmented['keypoints'], dtype=np.float32)

        if aug_keypoints.shape[0] == 0 or aug_keypoints.shape[0] % 4 != 0:
            # All bboxes were removed or invalid keypoint count
            if is_base_boxes:
                # Return empty QuadriBoxes to maintain type consistency
                results['gt_bboxes'] = QuadriBoxes(
                    torch.zeros((0, 8), dtype=torch.float32))
            else:
                results['gt_bboxes'] = np.zeros((0, 8), dtype=np.float32)
            results['gt_labels'] = np.array([], dtype=np.int64)
        else:
            # Reshape back to (M, 8) qbox format
            transformed_qboxes = aug_keypoints.reshape(-1, 8)
            num_boxes = transformed_qboxes.shape[0]

            if is_base_boxes:
                # Convert back to QuadriBoxes for ConvertBoxType.
                results['gt_bboxes'] = QuadriBoxes(
                    torch.from_numpy(transformed_qboxes).float())
            else:
                results['gt_bboxes'] = transformed_qboxes

            if label_array.shape[0] >= num_boxes:
                results['gt_labels'] = label_array[:num_boxes].astype(np.int64)
            else:
                padded_labels = np.zeros((num_boxes, ), dtype=np.int64)
                padded_labels[:label_array.shape[0]] = label_array.astype(
                    np.int64)
                results['gt_labels'] = padded_labels

        return results
