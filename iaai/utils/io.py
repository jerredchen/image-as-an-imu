import torch
import cv2
import pandas as pd
import numpy as np
from safetensors.torch import load_file

def to_tensor_func(arr):
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def read_img_path(path, resize=False, data_width=None, data_height=None, device='cuda'):
    img = cv2.imread(str(path))
    if resize:
        img = cv2.resize(img, (data_width, data_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_BCHW = to_tensor_func(img / 255.).float().to(device)
    return img_BCHW

def load_strayscanner_data(seq_dir):
    """Load StrayScanner data."""
    # Load IMU and rgb data
    imu_data_path = seq_dir / "imu.csv"
    frame_data_path = seq_dir / "odometry.csv"
    rgb_dir_path = seq_dir / "rgb_320x224"
    vel_data_path = seq_dir / "velocities.csv"
    opencv_poses_path = seq_dir / "opencv_poses.csv"
    
    imu_data = pd.read_csv(imu_data_path)
    frame_data = pd.read_csv(frame_data_path)
    vel_data = pd.read_csv(vel_data_path)
    opencv_poses_data = pd.read_csv(opencv_poses_path)
    
    # Clean column names
    frame_data.columns = frame_data.columns.str.strip()
    vel_data.columns = vel_data.columns.str.strip()
    imu_data.columns = imu_data.columns.str.strip()
    
    # Extract timestamps and other data
    frame_ts = frame_data['timestamp'].values
    imu_ts = imu_data['timestamp'].values
    arkit_ts = vel_data['timestamp'].values
    exposure_time_s = frame_data['exposure'].values
    gt_rots = opencv_poses_data[['qw', 'qx', 'qy', 'qz']].values
    gt_trans = opencv_poses_data[['t_x', 't_y', 't_z']].values

    rgb_files = list(rgb_dir_path.glob("*.jpg"))
    rgb_files.sort()

    # Ensure rgb_files and frame_data have the same length
    assert len(rgb_files) + 1 == len(frame_data)
    frame_data = frame_data[1:]
    frame_ts = frame_ts[1:]
    exposure_time_s = exposure_time_s[1:]

    # Load camera matrix
    cam_matrix_path = seq_dir / "camera_matrix_320x224.csv"
    with open(cam_matrix_path, "r") as f:
        cam_matrix = np.array([list(map(float, line.split(','))) for line in f])
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    intrinsic = [fx, fy, cx, cy]

    return {
        'rgb_files': rgb_files,
        'frame_ts': frame_ts,
        'imu_ts': imu_ts,
        'arkit_ts': arkit_ts,
        'exposure_time_s': exposure_time_s,
        'gt_rots': gt_rots,
        'gt_trans': gt_trans,
        'intrinsic': intrinsic,
        'imu_data': imu_data,
        'vel_data': vel_data,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }

def load_ckpt(ckpt_path):
    """Load checkpoint from either safetensors or regular torch format."""
    if str(ckpt_path).endswith('.safetensors'):
        return load_file(ckpt_path)
    else:
        return torch.load(ckpt_path)
