""" Get samples from Hypersim dataset
    Based on vkitti implementation
"""
import os
import cv2
import numpy as np

from infer.dataset_normal import Sample


def get_sample(base_data_dir, sample_path, info):
    # e.g. sample_path = "ai_001_001/rgb_cam_00_fr0000.png"
    scene_name = sample_path.split('/')[0]
    img_filename = sample_path.split('/')[1]
    
    # Extract frame number from filename like "rgb_cam_00_fr0000.png"
    frame_num = img_filename.split('_fr')[1].split('.')[0]
    
    dataset_path = os.path.join(base_data_dir, 'dsine_eval', 'hypersim')
    img_path = os.path.join(dataset_path, sample_path)
    
    # Build corresponding normal path
    normal_filename = f'depth_plane_cam_00_fr{frame_num}_1024x0768_normal_decoded_normal.png'
    normal_path = os.path.join(dataset_path, scene_name, normal_filename)
    
    assert os.path.exists(img_path), f"Image not found: {img_path}"
    assert os.path.exists(normal_path), f"Normal not found: {normal_path}"

    # read image (H, W, 3)
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    
    # read normal (H, W, 3)
    normal = cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    normal_mask = np.sum(normal, axis=2, keepdims=True) > 0
    normal = (normal.astype(np.float32) / 255.0) * 2.0 - 1.0

    # Create default intrinsics since hypersim doesn't provide them
    # Using typical values for 1024x768 resolution
    H, W = img.shape[:2]
    fx = fy = 0.8 * W  # Typical focal length assumption
    cx = W / 2.0
    cy = H / 2.0
    intrins = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # Extract img_name for compatibility
    img_name = img_filename.replace('.png', '')

    sample = Sample(
        img=img,
        normal=normal,
        normal_mask=normal_mask,
        intrins=intrins,

        dataset_name='hypersim',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample 