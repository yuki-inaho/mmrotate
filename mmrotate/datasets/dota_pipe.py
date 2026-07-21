# mmcv_custom_config/mmrotate/mmrotate/datasets/dota_pipe.py
import glob
import os.path as osp
from typing import List, Tuple

from mmengine.dataset import BaseDataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class DOTAPipeDataset(BaseDataset):
    """Dataset for pipe detection in DOTA format (RGB only)."""

    METAINFO = {
        "classes": ("pipe",),
        "palette": [(255, 165, 0)],  # Example color: Orange
    }

    def __init__(
        self,
        img_shape: Tuple[int, int] = (736, 512),  # Default H, W
        diff_thr: int = 100,  # DOTA difficulty threshold
        img_suffix=".jpg",  # Default image suffix
        **kwargs,
    ) -> None:
        self.img_shape = img_shape  # (height, width)
        self.diff_thr = diff_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        cls_map = {c: i for i, c in enumerate(self.metainfo["classes"])}
        data_list = []

        # ann_file is the path to the directory containing label txt files
        if not self.ann_file:  # For test mode where no annotation file is provided
            img_files = glob.glob(osp.join(self.data_prefix["img_path"], f"*{self.img_suffix}"))
            for img_path in img_files:
                data_info = {
                    "img_path": img_path,
                    "file_name": osp.basename(img_path),
                    "img_id": osp.splitext(osp.basename(img_path))[0],
                    "height": self.img_shape[0],
                    "width": self.img_shape[1],
                    "instances": [dict(bbox=[], bbox_label=[], ignore_flag=0)],  # Dummy instance
                }
                data_list.append(data_info)
            return data_list

        txt_files = glob.glob(osp.join(self.ann_file, "*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt annotation files found in {self.ann_file}")

        for txt_file in txt_files:
            img_id = osp.splitext(osp.basename(txt_file))[0]
            data_info = {
                "img_id": img_id,
                "file_name": f"{img_id}{self.img_suffix}",
                "img_path": osp.join(self.data_prefix["img_path"], f"{img_id}{self.img_suffix}"),
                "height": self.img_shape[0],
                "width": self.img_shape[1],
            }

            instances = []
            with open(txt_file) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue  # 8 coords + class_name [+ difficulty]

                    instance = {}
                    try:
                        instance["bbox"] = [float(p) for p in parts[:8]]
                    except ValueError:
                        print(f"Warning: Could not parse bbox in {txt_file}: {parts[:8]}")
                        continue

                    cls_name = parts[8].lower()  # Ensure lowercase if class names are lowercase
                    if cls_name not in cls_map:
                        # print(f"Warning: Class '{cls_name}' in {txt_file} not in METAINFO. Skipping.")
                        continue  # Skip if class name is not in defined classes
                    instance["bbox_label"] = cls_map[cls_name]

                    difficulty = int(parts[9]) if len(parts) > 9 else 0
                    instance["ignore_flag"] = 1 if difficulty > self.diff_thr else 0
                    instances.append(instance)

            data_info["instances"] = instances
            data_list.append(data_info)

        return data_list

    def filter_data(self) -> List[dict]:
        if self.test_mode:
            return self.data_list

        # Filter out images with no valid annotations if specified
        if self.filter_cfg and self.filter_cfg.get("filter_empty_gt", False):
            filtered_list = []
            for data_info in self.data_list:
                if any(inst["ignore_flag"] == 0 for inst in data_info.get("instances", [])):
                    filtered_list.append(data_info)
            return filtered_list
        return self.data_list

    def get_cat_ids(self, idx: int) -> List[int]:
        instances = self.get_data_info(idx)["instances"]
        return [inst["bbox_label"] for inst in instances if inst.get("ignore_flag", 0) == 0]
