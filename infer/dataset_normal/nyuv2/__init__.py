""" Get samples from NYUv2 (https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
    NOTE: GT surface normals are from GeoNet (CVPR 2018) - https://github.com/xjqi/GeoNet
"""
import os
import cv2
import numpy as np

from infer.dataset_normal import Sample


def get_sample(base_data_dir, sample_path, info):
    # e.g. sample_path = "test/000000_img.png"
    scene_name = sample_path.split('/')[0]
    img_name, img_ext = sample_path.split('/')[1].split('_img')

    dataset_path = os.path.join(base_data_dir, 'dsine_eval', 'nyuv2')
    img_path = '%s/%s' % (dataset_path, sample_path)
    normal_png_path = img_path.replace('_img'+img_ext, '_normal.png')
    normal_npy_path = img_path.replace('_img'+img_ext, '_normal.npy')
    intrins_path = img_path.replace('_img'+img_ext, '_intrins.npy')
    assert os.path.exists(img_path)

    # read image (H, W, 3)
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    #保存图像
    # cv2.imwrite(os.path.join(base_data_dir, img_name+'_img.png'), img*255)
    
    # read normal (H, W, 3)
    if os.path.exists(normal_png_path):
        normal = cv2.cvtColor(cv2.imread(normal_png_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        normal_mask = np.sum(normal, axis=2, keepdims=True) > 0
        normal = (normal.astype(np.float32) / 255.0) * 2.0 - 1.0
    elif os.path.exists(normal_npy_path):
        normal = np.load(normal_npy_path).astype(np.float32)
        assert normal.ndim == 3 and normal.shape[2] == 3, f"Unexpected normal shape: {normal.shape}"
        # GeoNet npy normals use opposite x-axis convention for this evaluation codepath.
        normal[:, :, 0] *= -1.0
        normal_mask = np.linalg.norm(normal, axis=2, keepdims=True) > 1e-6
    else:
        raise FileNotFoundError(f"Missing NYUv2 normal file: {normal_png_path} or {normal_npy_path}")

    # read intrins (3, 3)
    if os.path.exists(intrins_path):
        intrins = np.load(intrins_path)
    else:
        # Fallback to NYUv2 default intrinsics used by many benchmarks.
        intrins = np.array([
            [518.8579, 0.0, 325.5824],
            [0.0, 519.4696, 253.7362],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    sample = Sample(
        img=img,
        normal=normal,
        normal_mask=normal_mask,
        intrins=intrins,

        dataset_name='nyuv2',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample
