import os

from .base_depth_dataset import BaseDepthDataset, get_pred_name, DatasetMode  # noqa: F401
from .diode_dataset import DIODEDataset
from .eth3d_dataset import ETH3DDataset
from .kitti_dataset import KITTIDataset
from .nyu_dataset import NYUDataset
from .scannet_dataset import ScanNetDataset


dataset_name_class_dict = {
    "nyu_v2": NYUDataset,
    "kitti": KITTIDataset,
    "eth3d": ETH3DDataset,
    "diode": DIODEDataset,
    "scannet": ScanNetDataset,
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_split_file(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path

    candidate = os.path.join(REPO_ROOT, path)
    if os.path.exists(candidate):
        return candidate

    return path


def get_dataset(cfg_data_split, base_data_dir: str, mode: DatasetMode, prompt_type="query", **kwargs) -> BaseDepthDataset:
    if cfg_data_split.name in dataset_name_class_dict.keys():
        dataset_class = dataset_name_class_dict[cfg_data_split.name]
        filename_ls_path = cfg_data_split.filenames if not prompt_type == "full" else (cfg_data_split.filenames).replace(".txt", "_wc.txt")
        filename_ls_path = _resolve_split_file(filename_ls_path)
        dataset = dataset_class(
            mode=mode,
            filename_ls_path=filename_ls_path,
            dataset_dir=os.path.join(base_data_dir, cfg_data_split.dir),
            **cfg_data_split,
            prompt_type=prompt_type,
            **kwargs,
        )
    else:
        raise NotImplementedError

    return dataset
