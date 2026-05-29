"""
FenceMask Transform for MMRotate

This transform creates fence-like occlusion patterns to improve model robustness.
The fence pattern consists of vertical or horizontal lines that occlude parts of the image.
"""

import cv2
import numpy as np
import random
from typing import Dict, Optional, Union, Tuple, Any

from mmrotate.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class FenceMask(BaseTransform):
    """
    Apply fence-like occlusion patterns to the image.
    
    This transform creates fence-like patterns by drawing parallel lines
    across the image, simulating occlusion effects to improve model robustness.
    
    Args:
        fence_direction (str): Direction of fence lines. Either 'vertical' or 'horizontal'.
            Default: 'vertical'.
        line_width (int or tuple): Width of fence lines in pixels. If tuple, randomly
            select from range. Default: (2, 8).
        line_spacing (int or tuple): Spacing between fence lines in pixels. If tuple,
            randomly select from range. Default: (20, 60).
        fence_opacity (float or tuple): Opacity of fence lines (0.0 to 1.0). If tuple,
            randomly select from range. Default: (0.3, 0.8).
        fence_color (tuple): RGB color of fence lines. Default: (0, 0, 0) for black.
        coverage_ratio (float or tuple): Ratio of image to be covered by fence.
            If tuple, randomly select from range. Default: (0.1, 0.4).
        prob (float): Probability of applying this transform. Default: 0.5.
    """
    
    def __init__(self,
                 fence_direction: str = 'vertical',
                 line_width: Union[int, Tuple[int, int]] = (2, 8),
                 line_spacing: Union[int, Tuple[int, int]] = (20, 60),
                 fence_opacity: Union[float, Tuple[float, float]] = (0.3, 0.8),
                 fence_color: Tuple[int, int, int] = (0, 0, 0),
                 coverage_ratio: Union[float, Tuple[float, float]] = (0.1, 0.4),
                 prob: float = 0.5):
        
        self.fence_direction = fence_direction
        self.line_width = line_width
        self.line_spacing = line_spacing
        self.fence_opacity = fence_opacity
        self.fence_color = fence_color
        self.coverage_ratio = coverage_ratio
        self.prob = prob
        
        assert fence_direction in ['vertical', 'horizontal'], \
            f"fence_direction must be 'vertical' or 'horizontal', got {fence_direction}"
        assert 0.0 <= prob <= 1.0, f"prob must be in [0.0, 1.0], got {prob}"
    
    def _get_random_value(self, param: Union[int, float, Tuple]) -> Union[int, float]:
        """Get random value from parameter."""
        if isinstance(param, (int, float)):
            return param
        elif isinstance(param, tuple) and len(param) == 2:
            if isinstance(param[0], int):
                return random.randint(param[0], param[1])
            else:
                return random.uniform(param[0], param[1])
        else:
            raise ValueError(f"Invalid parameter format: {param}")
    
    def _apply_fence_mask(self, img: np.ndarray) -> np.ndarray:
        """Apply fence mask to image."""
        h, w = img.shape[:2]
        
        # Get random parameters
        line_width = self._get_random_value(self.line_width)
        line_spacing = self._get_random_value(self.line_spacing)
        fence_opacity = self._get_random_value(self.fence_opacity)
        coverage_ratio = self._get_random_value(self.coverage_ratio)
        
        # Create fence mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if self.fence_direction == 'vertical':
            # Vertical fence lines
            total_width = int(w * coverage_ratio)
            start_x = random.randint(0, max(1, w - total_width))
            
            x = start_x
            while x < start_x + total_width and x < w:
                x_end = min(x + line_width, w)
                mask[:, x:x_end] = 255
                x += line_width + line_spacing
                
        else:  # horizontal
            # Horizontal fence lines
            total_height = int(h * coverage_ratio)
            start_y = random.randint(0, max(1, h - total_height))
            
            y = start_y
            while y < start_y + total_height and y < h:
                y_end = min(y + line_width, h)
                mask[y:y_end, :] = 255
                y += line_width + line_spacing
        
        # Apply fence mask to image
        result = img.copy()
        fence_mask = mask > 0
        
        # Blend fence color with original image
        for c in range(img.shape[2]):
            result[fence_mask, c] = (
                (1 - fence_opacity) * img[fence_mask, c] + 
                fence_opacity * self.fence_color[c]
            ).astype(np.uint8)
        
        return result
    
    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply FenceMask transform to results."""
        if random.random() > self.prob:
            return results
        
        img = results['img']
        
        # Apply fence mask
        results['img'] = self._apply_fence_mask(img)
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(fence_direction={self.fence_direction}, '
        repr_str += f'line_width={self.line_width}, '
        repr_str += f'line_spacing={self.line_spacing}, '
        repr_str += f'fence_opacity={self.fence_opacity}, '
        repr_str += f'fence_color={self.fence_color}, '
        repr_str += f'coverage_ratio={self.coverage_ratio}, '
        repr_str += f'prob={self.prob})'
        return repr_str