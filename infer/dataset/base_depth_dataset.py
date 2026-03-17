import io
import os
import random
import tarfile
from enum import Enum

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize
import torchvision.transforms.functional as F

class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"


def read_image_from_tar(tar_obj, img_rel_path):
    image = tar_obj.extractfile("./" + img_rel_path)
    image = image.read()
    image = Image.open(io.BytesIO(image))


class BaseDepthDataset(Dataset):

    def __init__(
        self, mode: DatasetMode, filename_ls_path: str,        dataset_dir: str,        disp_name: str,        min_depth,        max_depth,        has_filled_depth,        name_mode,        depth_transform=None,        augmentation_args: dict = None,        resize_to_hw=None,
        move_invalid_to_far_plane: bool = True, rgb_transform=None, prompt_type="query", **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode

        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        self.disp_name = disp_name
        self.has_filled_depth = has_filled_depth
        self.name_mode: DepthFileNameMode = name_mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.depth_transform = depth_transform
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.prompt_type = prompt_type
        # 设置默认的rgb_transform函数
        if rgb_transform is None:
            self.rgb_transform = self._default_rgb_transform
        else:
            self.rgb_transform = rgb_transform
        self.move_invalid_to_far_plane = move_invalid_to_far_plane

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [s.split() for s in f.readlines()]  # [['rgb.png', 'depth.tif'], [], ...]

        # Tar dataset
        self.tar_obj = None
        self.tar_obj_pid = None
        self.is_tar = (True if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir) else False)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        rgb_rel_path, depth_rel_path, filled_rel_path, prompt = self._get_data_path(index=index)

        rasters = {}

        # RGB data
        rasters.update(self._load_rgb_data(rgb_rel_path=rgb_rel_path))

        # Depth data
        if DatasetMode.RGB_ONLY != self.mode:
            # load data
            depth_data = self._load_depth_data(depth_rel_path=depth_rel_path, filled_rel_path=filled_rel_path)
            rasters.update(depth_data)
            # valid mask
            rasters["valid_mask_raw"] = self._get_valid_mask(rasters["depth_raw_linear"]).clone()
            rasters["valid_mask_filled"] = self._get_valid_mask(rasters["depth_filled_linear"]).clone()

        
        other = {"index": index, "rgb_relative_path": rgb_rel_path, "prompt": prompt}

        return rasters, other

    def _load_rgb_data(self, rgb_rel_path):
        # Read RGB data
        _, rgb = self._read_image(rgb_rel_path)
        rgb = self.input_process_image(rgb)
        outputs = {"rgb": rgb}
        return outputs

    def input_process_image(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            return image
        elif isinstance(image, Image.Image):
            image = F.to_tensor(image.convert("RGB"))
            
            return image
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, str):
            image = F.to_tensor(Image.open(image).convert("RGB"))
            
            return image
        return image

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        # Read depth data
        outputs = {}
        depth_raw = self._read_depth_file(depth_rel_path).squeeze()
        depth_raw_linear = torch.from_numpy(depth_raw).float().unsqueeze(0)  # [1, H, W]
        outputs["depth_raw_linear"] = depth_raw_linear.clone()

        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled).float().unsqueeze(0)
            outputs["depth_filled_linear"] = depth_filled_linear
        else:
            outputs["depth_filled_linear"] = depth_raw_linear.clone()

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]
        rgb_rel_path = filename_line[0]
        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = filename_line[1]
            if self.has_filled_depth:
                filled_rel_path = filename_line[2]
                
        if self.prompt_type == "full":
            if filename_line[2][0] in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
                prompt = ' '.join(filename_line[2:])
            else:
                prompt = ' '.join(filename_line[3:])
        else:
            prompt = 1
        return rgb_rel_path, depth_rel_path, filled_rel_path, prompt

    def _read_image(self, img_rel_path):
        if self.is_tar:
            tar_obj = self._ensure_tar_obj()
            image = tar_obj.extractfile("./" + img_rel_path)
            image = image.read()
            image = Image.open(io.BytesIO(image))
        else:
            img_path = os.path.join(self.dataset_dir, img_rel_path)
            image = Image.open(img_path)
        image_arr = np.asarray(image)
        return image_arr, image

    def _read_depth_file(self, rel_path):
        depth_in, _ = self._read_image(rel_path)
        #  Replace code below to decode depth according to dataset definition
        depth_decoded = depth_in

        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and((depth > self.min_depth), (depth < self.max_depth)).bool()
        return valid_mask

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # Normalization
        rasters["depth_raw_norm"] = self.depth_transform(rasters["depth_raw_linear"], rasters["valid_mask_raw"]).clone()
        rasters["depth_filled_norm"] = self.depth_transform(rasters["depth_filled_linear"], rasters["valid_mask_filled"]).clone()

        # Set invalid pixel to far plane
        if self.move_invalid_to_far_plane:
            if self.depth_transform.far_plane_at_max:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = self.depth_transform.norm_max
            else:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = self.depth_transform.norm_min

        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT)
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters

    def _augment_data(self, rasters_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}

        return rasters_dict

    def __del__(self):
        if self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None
            self.tar_obj_pid = None

    def _default_rgb_transform(self, x):
        """默认的RGB变换函数: [0, 255] -> [-1, 1]"""
        return x / 255.0 * 2 - 1

    def _ensure_tar_obj(self):
        """Ensure each process owns its own tar handle to avoid cross-process FD issues."""
        if not self.is_tar:
            return None
        current_pid = os.getpid()
        if self.tar_obj is None or self.tar_obj_pid != current_pid:
            if self.tar_obj is not None:
                try:
                    self.tar_obj.close()
                except Exception:
                    pass
            self.tar_obj = tarfile.open(self.dataset_dir)
            self.tar_obj_pid = current_pid
        return self.tar_obj


# Prediction file naming modes
class DepthFileNameMode(Enum):
    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if DepthFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif DepthFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif DepthFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif DepthFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename
