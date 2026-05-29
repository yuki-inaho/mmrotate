"""
RandomLines Transform for MMRotate

This transform creates random line occlusion patterns to improve model robustness.
The lines can have various angles, thicknesses, and colors to simulate real-world occlusions.
"""

import cv2
import numpy as np
import random
import math
from typing import Dict, Optional, Union, Tuple, Any, List

from mmrotate.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class RandomLines(BaseTransform):
    """
    Apply random line occlusion patterns to the image.

    This transform creates random lines across the image with varying angles,
    thicknesses, and colors to simulate occlusion effects and improve model robustness.
    
    Args:
        num_lines (int or tuple): Number of lines to draw. If tuple, randomly
            select from range. Default: (1, 5).
        line_thickness (int or tuple): Thickness of lines in pixels. If tuple,
            randomly select from range. Default: (1, 8).
        line_length_ratio (float or tuple): Ratio of line length to image diagonal.
            If tuple, randomly select from range. Default: (0.1, 0.8).
        line_opacity (float or tuple): Opacity of lines (0.0 to 1.0). If tuple,
            randomly select from range. Default: (0.3, 0.9).
        line_colors (list): List of possible RGB colors for lines. Random color
            will be selected from this list. Default: [(0, 0, 0), (255, 255, 255)].
        angle_range (tuple): Range of line angles in degrees. Default: (0, 180).
        prob (float): Probability of applying this transform. Default: 0.5.
    """
    
    def __init__(self,
                 num_lines: Union[int, Tuple[int, int]] = (1, 5),
                 line_thickness: Union[int, Tuple[int, int]] = (1, 8),
                 line_length_ratio: Union[float, Tuple[float, float]] = (0.1, 0.8),
                 line_opacity: Union[float, Tuple[float, float]] = (0.3, 0.9),
                 line_colors: List[Tuple[int, int, int]] = [(0, 0, 0), (255, 255, 255)],
                 angle_range: Tuple[float, float] = (0, 180),
                 prob: float = 0.5):
        
        self.num_lines = num_lines
        self.line_thickness = line_thickness
        self.line_length_ratio = line_length_ratio
        self.line_opacity = line_opacity
        self.line_colors = line_colors
        self.angle_range = angle_range
        self.prob = prob
        
        assert 0.0 <= prob <= 1.0, f"prob must be in [0.0, 1.0], got {prob}"
        assert len(line_colors) > 0, "line_colors must not be empty"
        assert len(angle_range) == 2, "angle_range must be a tuple of length 2"
    
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
    
    def _generate_random_line(self, img_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Generate parameters for a random line."""
        h, w = img_shape[:2]
        diagonal = math.sqrt(h*h + w*w)
        
        # Get random parameters
        thickness = int(self._get_random_value(self.line_thickness))
        length_ratio = self._get_random_value(self.line_length_ratio)
        opacity = self._get_random_value(self.line_opacity)
        color = random.choice(self.line_colors)
        angle = random.uniform(self.angle_range[0], self.angle_range[1])
        
        # Calculate line length
        line_length = int(diagonal * length_ratio)
        
        # Generate random start point
        start_x = random.randint(0, w-1)
        start_y = random.randint(0, h-1)
        
        # Calculate end point based on angle and length
        angle_rad = math.radians(angle)
        end_x = int(start_x + line_length * math.cos(angle_rad))
        end_y = int(start_y + line_length * math.sin(angle_rad))
        
        return {
            'start_point': (start_x, start_y),
            'end_point': (end_x, end_y),
            'thickness': thickness,
            'opacity': opacity,
            'color': color
        }
    
    def _draw_line_with_opacity(self, img: np.ndarray, line_params: Dict[str, Any]) -> np.ndarray:
        """Draw a line on the image with specified opacity."""
        result = img.copy()
        h, w = img.shape[:2]
        
        # Create a mask for the line
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw line on mask
        cv2.line(mask, 
                line_params['start_point'], 
                line_params['end_point'], 
                255, 
                line_params['thickness'])
        
        # Apply opacity blending
        line_mask = mask > 0
        opacity = line_params['opacity']
        color = line_params['color']
        
        for c in range(img.shape[2]):
            result[line_mask, c] = (
                (1 - opacity) * img[line_mask, c] + 
                opacity * color[c]
            ).astype(np.uint8)
        
        return result
    
    def _apply_random_lines(self, img: np.ndarray) -> np.ndarray:
        """Apply random lines to image."""
        result = img.copy()
        
        # Get number of lines to draw
        num_lines = self._get_random_value(self.num_lines)
        
        # Draw random lines
        for _ in range(num_lines):
            line_params = self._generate_random_line(img.shape)
            result = self._draw_line_with_opacity(result, line_params)
        
        return result
    
    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply RandomLines transform to results."""
        if random.random() > self.prob:
            return results
        
        img = results['img']
        
        # Apply random lines
        results['img'] = self._apply_random_lines(img)
        
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(num_lines={self.num_lines}, '
        repr_str += f'line_thickness={self.line_thickness}, '
        repr_str += f'line_length_ratio={self.line_length_ratio}, '
        repr_str += f'line_opacity={self.line_opacity}, '
        repr_str += f'line_colors={self.line_colors}, '
        repr_str += f'angle_range={self.angle_range}, '
        repr_str += f'prob={self.prob})'
        return repr_str