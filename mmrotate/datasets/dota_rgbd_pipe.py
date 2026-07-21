# mmcv_custom_config/mmrotate/mmrotate/datasets/dota_rgbd_pipe.py
import glob
import os.path as osp
import logging
from typing import List
from mmengine.dataset import BaseDataset
from mmengine.logging import print_log  # ログ出力用
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DOTARGBDPipeDataset(BaseDataset):
    """
    DOTA-like dataset for RGB-D Pipe detection.

    This dataset class is designed for a DOTA-like annotation format where
    annotations are provided in text files, one per image. It specifically
    handles RGB and Depth image pairs, where filenames are derived from
    the annotation's base name with specific suffixes for RGB and Depth images.

    Args:
        rgb_suffix (str): Suffix for RGB image files.
            Defaults to '_rgb.jpg'.
        depth_suffix (str): Suffix for depth map files.
            Defaults to '_depth.png'.
        diff_thr (int): Difficulty threshold for ground truth. Bboxes with
            difficulty higher than this will be marked as ignored.
            Defaults to 100.
        kwargs: Other arguments are passed to the `BaseDataset` superclass.
    """

    METAINFO = {
        "classes": ("pipe",),  # Example, should be configured
        "palette": [(220, 20, 60)],
    }

    def __init__(self, rgb_suffix: str = "_rgb.jpg", depth_suffix: str = "_depth.png", diff_thr: int = 100, **kwargs):
        self.rgb_suffix = rgb_suffix
        self.depth_suffix = depth_suffix
        self.diff_thr = diff_thr

        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotations and prepare the list of data items.

        Returns:
            List[dict]: A list of data information dicts.
        """
        if "classes" not in self.metainfo:
            # This check might be redundant if BaseDataset already handles it.
            # However, it's a good safeguard for custom datasets.
            if self.METAINFO and "classes" in self.METAINFO:
                self.metainfo["classes"] = self.METAINFO["classes"]
                print_log(f"Using 'classes' from METAINFO: {self.metainfo['classes']}", logger="current", level=logging.INFO)
            else:
                raise ValueError(
                    "Classes not found in dataset metainfo. "
                    "Please set 'classes' in 'metainfo' argument or "
                    "define a METAINFO dict with 'classes' in the dataset class."
                )

        cls_map = {c: i for i, c in enumerate(self.metainfo["classes"])}
        data_list = []

        # self.ann_file should be the absolute path to the annotation directory
        # as resolved by BaseDataset's __init__ (data_root + ann_file from config)
        ann_dir_path = self.ann_file

        if not osp.isdir(ann_dir_path):
            raise ValueError(
                f"Annotation directory '{ann_dir_path}' not found or not a directory. "
                f"Ensure 'ann_file' in your config correctly points to the "
                f"directory containing annotation text files (relative to 'data_root')."
            )

        txt_files = glob.glob(osp.join(ann_dir_path, "*.txt"))

        if not txt_files:
            print_log(
                f"No annotation .txt files found in '{ann_dir_path}'. The dataset will be empty.",
                logger="current",
                level=logging.WARNING,
            )
            return []

        num_skipped_due_to_missing_rgb = 0
        num_skipped_due_to_missing_depth = 0
        num_skipped_due_to_parsing_error = 0
        num_skipped_due_to_empty_gt = 0

        for txt_file in txt_files:
            data_info = {}
            base_name = osp.splitext(osp.basename(txt_file))[0]
            data_info["img_id"] = base_name

            if (
                not isinstance(self.data_prefix, dict)
                or "img_path" not in self.data_prefix
                or "depth_map_path" not in self.data_prefix
            ):
                raise ValueError(
                    "`data_prefix` must be a dict with 'img_path' "
                    "and 'depth_map_path' keys, specifying subdirectories "
                    "relative to 'data_root'."
                )

            rgb_img_name = base_name + self.rgb_suffix
            depth_map_name = base_name + self.depth_suffix

            # Construct absolute paths for image and depth map
            # self.data_root is the absolute path to the dataset's root directory
            current_img_dir = osp.join(self.data_root, self.data_prefix["img_path"])
            current_depth_dir = osp.join(self.data_root, self.data_prefix["depth_map_path"])

            data_info["img_path"] = osp.join(current_img_dir, rgb_img_name)
            data_info["depth_map_path"] = osp.join(current_depth_dir, depth_map_name)

            # Check file existence early to provide clearer error messages
            if not osp.exists(data_info["img_path"]):
                print_log(
                    f"RGB image file not found: {data_info['img_path']} (derived from annotation: {txt_file})",
                    logger="current",
                    level=logging.WARNING,
                )
                num_skipped_due_to_missing_rgb += 1
                continue
            if not osp.exists(data_info["depth_map_path"]):
                print_log(
                    f"Depth map file not found: {data_info['depth_map_path']} (derived from annotation: {txt_file})",
                    logger="current",
                    level=logging.WARNING,
                )
                num_skipped_due_to_missing_depth += 1
                continue

            instances = []
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                    # Check for empty annotation file *before* iterating lines
                    # filter_cfg is available as self.filter_cfg
                    filter_empty = True  # Default to True for training
                    if self.filter_cfg is not None:
                        filter_empty = self.filter_cfg.get("filter_empty_gt", True)

                    if not lines and filter_empty and not self.test_mode:
                        num_skipped_due_to_empty_gt += 1
                        continue

                    for line_idx, si in enumerate(lines):
                        instance = {}
                        bbox_info = si.strip().split()

                        if len(bbox_info) < 9:  # 8 coords + class_name
                            print_log(
                                f"Malformed line {line_idx + 1} in {txt_file}: '{si.strip()}' "
                                f"- expected at least 9 fields. Skipping this line.",
                                logger="current",
                                level=logging.WARNING,
                            )
                            continue
                        try:
                            instance["bbox"] = [float(i) for i in bbox_info[:8]]
                        except ValueError:
                            print_log(
                                f"Malformed bbox coordinates in line {line_idx + 1} of {txt_file}: "
                                f"'{bbox_info[:8]}'. Skipping this line.",
                                logger="current",
                                level=logging.WARNING,
                            )
                            continue

                        cls_name = bbox_info[8].lower()
                        if cls_name not in cls_map:
                            print_log(
                                f"Unknown class name '{cls_name}' in line {line_idx + 1} of {txt_file}. "
                                f"Available classes: {list(cls_map.keys())}. Skipping this instance.",
                                logger="current",
                                level=logging.WARNING,
                            )
                            continue
                        instance["bbox_label"] = cls_map[cls_name]

                        difficulty = 0
                        if len(bbox_info) >= 10:
                            try:
                                difficulty = int(bbox_info[9])
                            except ValueError:
                                print_log(
                                    f"Malformed difficulty '{bbox_info[9]}' in line {line_idx + 1} of {txt_file}. "
                                    f"Using default difficulty 0.",
                                    logger="current",
                                    level=logging.WARNING,
                                )
                        instance["difficulty"] = difficulty  # Store for potential use

                        if difficulty > self.diff_thr:
                            instance["ignore_flag"] = 1
                        else:
                            instance["ignore_flag"] = 0
                        instances.append(instance)
            except Exception as e:
                print_log(f"Error reading or parsing annotation file {txt_file}: {e}", logger="current", level=logging.ERROR)
                num_skipped_due_to_parsing_error += 1
                continue

            # Second check for empty instances after parsing lines,
            # if filter_empty_gt is True and not in test_mode.
            if not instances and filter_empty and not self.test_mode:
                num_skipped_due_to_empty_gt += 1
                continue

            data_info["instances"] = instances
            # `height` and `width` are not required here as they are added by the loading pipeline.
            # If `bbox_min_size` filter is used in `filter_cfg`, `height` and `width` would be needed.
            # However, for DOTA-like datasets, images are often pre-split, so original image size
            # might not be directly relevant for filtering individual annotations this way.
            # If needed, LoadRGBDImageFromFile could add 'ori_shape' earlier.
            data_list.append(data_info)

        # Log summary of skipped files
        if num_skipped_due_to_missing_rgb > 0:
            print_log(
                f"Total skipped {num_skipped_due_to_missing_rgb} samples due to missing RGB images.",
                logger="current",
                level=logging.WARNING,
            )
        if num_skipped_due_to_missing_depth > 0:
            print_log(
                f"Total skipped {num_skipped_due_to_missing_depth} samples due to missing depth maps.",
                logger="current",
                level=logging.WARNING,
            )
        if num_skipped_due_to_parsing_error > 0:
            print_log(
                f"Total skipped {num_skipped_due_to_parsing_error} annotation files due to parsing errors.",
                logger="current",
                level=logging.WARNING,
            )
        if num_skipped_due_to_empty_gt > 0:
            print_log(
                f"Total skipped {num_skipped_due_to_empty_gt} samples due to empty ground truth (and filter_empty_gt=True).",
                logger="current",
                level=logging.WARNING,
            )

        if not data_list:
            print_log(
                f"The final data_list is empty. This will likely cause errors downstream. "
                f"Please check dataset paths, file formats, and `filter_cfg` in your config. "
                f"Annotation directory was: '{ann_dir_path}'",
                logger="current",
                level=logging.ERROR,
            )

        return data_list

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index. This is required for COCO-style evaluation.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        # self.data_list is populated by load_data_list and potentially filtered by filter_data
        # self.get_data_info(idx) is the recommended way to access an item from BaseDataset
        data_info = self.get_data_info(idx)

        if "instances" not in data_info or not data_info["instances"]:
            # This can happen if filter_empty_gt=False and an image has no annotations,
            # or if in test_mode with no annotations.
            return []

        # 'bbox_label' should be the integer index of the class
        return [instance["bbox_label"] for instance in data_info["instances"]]

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return (
            f"{self.__class__.__name__}("
            f"data_root={self.data_root}, "
            f"ann_file={self.ann_file}, "
            f"data_prefix={self.data_prefix}, "
            f"rgb_suffix={self.rgb_suffix}, "
            f"depth_suffix={self.depth_suffix}, "
            f"diff_thr={self.diff_thr}, "
            f"metainfo={self.metainfo}, "
            f"filter_cfg={self.filter_cfg}, "
            # `pipeline` can be very long, so maybe omit or summarize
            # f'pipeline={self.pipeline}, '
            f"test_mode={self.test_mode})"
        )
