import collections
import json
import sys
import os
import cv2
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'ECCV2022-RIFE')))
from train_log.RIFE_HDv3 import Model as RIFEModel
from iaai.utils.io import read_img_path

def invert_Rt(R, T):
    """Inverts a pose to go from camera-to-world extrinsics to world-to-camera pose."""
    R_inv = R.T
    T_inv = -R_inv @ T
    return R_inv, T_inv

def invert_pose(pose_44):
    """Inverts a 4x4 pose matrix."""
    R = pose_44[:3, :3]
    T = pose_44[:3, 3]
    R_inv, T_inv = invert_Rt(R, T)
    pose_inv = torch.eye(4).float()
    pose_inv[:3, :3] = R_inv
    pose_inv[:3, 3] = T_inv
    return pose_inv

################################################################################
# Functions for reading COLMAP data
# From: https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_images_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_cameras_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

################################################################################

################################################################################
# I/O utility functions from PromptDA
# Used and adapted from: https://github.com/DepthAnything/PromptDA/blob/main/promptda/utils/io_wrapper.py
def to_tensor_func(arr):
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def to_numpy_func(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr

def save_depth(depth_tensor, path):
    depth_map = to_numpy_func(depth_tensor)
    depth_map = (depth_map * 1000).astype(np.uint16)
    imageio.imwrite(path, depth_map)

################################################################################

def read_arkit_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_camera_params(cam_id, cam_data, data_width, data_height):
    fx_og_px, fy_og_px = cam_data[cam_id].params[:2]
    cx_og_px, cy_og_px = cam_data[cam_id].params[2:4]
    height_og, width_og = cam_data[cam_id].height, cam_data[cam_id].width
    scale_factor = max(data_height / height_og, data_width / width_og)
    fx_px = fx_og_px * scale_factor
    fy_px = fy_og_px * scale_factor
    cx = cx_og_px * scale_factor
    cy = cy_og_px * scale_factor
    K = torch.tensor([[fx_px, 0, cx], [0, fy_px, cy], [0, 0, 1]], dtype=torch.float32).cuda()
    return K

def calculate_cos_angle(v1, v2):
    cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
    return cos_angle

def detect_arkit_pose_jump(
        pose_intr_imu: dict,
        img1_stem: str,
        frame_num_gap: int,
        norm_ratio_thresh=3.0,
        ab_bc_angle_thresh=0.8,
        bc_cd_angle_thresh=0.8,
        ab_cd_angle_thresh=0.9
    ) -> bool:
    jump_detected = False
    img1_frame_num = int(img1_stem.split('_')[-1])
    prev_stem = "frame_" + str(img1_frame_num - 1).zfill(6)
    mid1_stem = img1_stem
    for idx in range(1, frame_num_gap+1):
        mid2_stem = "frame_" + str(img1_frame_num + idx).zfill(6)
        next_stem = "frame_" + str(img1_frame_num + idx + 1).zfill(6)
        if (mid2_stem not in pose_intr_imu) or (next_stem not in pose_intr_imu):
            break
        wTa = torch.tensor(pose_intr_imu[prev_stem]['aligned_pose'])
        wTb = torch.tensor(pose_intr_imu[mid1_stem]['aligned_pose'])
        wTc = torch.tensor(pose_intr_imu[mid2_stem]['aligned_pose'])
        wTd = torch.tensor(pose_intr_imu[next_stem]['aligned_pose'])
        vec_ab = wTb[:3, 3] - wTa[:3, 3]
        vec_bc = wTc[:3, 3] - wTb[:3, 3]
        vec_cd = wTd[:3, 3] - wTc[:3, 3]
        # Check relative magnitude and angle between vectors
        if np.linalg.norm(vec_bc) / np.linalg.norm(vec_ab) >= norm_ratio_thresh or \
            (abs(calculate_cos_angle(vec_ab, vec_bc)) <= ab_bc_angle_thresh and \
                abs(calculate_cos_angle(vec_bc, vec_cd)) <= bc_cd_angle_thresh and \
                abs(calculate_cos_angle(vec_ab, vec_cd)) >= ab_cd_angle_thresh):
                jump_detected = True
                break
        else:
            prev_stem = mid1_stem
            mid1_stem = mid2_stem
    return jump_detected

def backproject(depth_map, fl_px, pp_px):
    """Backprojects a depth map into 3D space using intrinsics."""
    h, w = depth_map.shape[-2:]
    cx, cy = pp_px

    # Generate homogeneous pixel coordinates
    u, v = torch.meshgrid(
        torch.arange(w),
        torch.arange(h),
        indexing='xy'
    )
    u = (u.to(depth_map.device).float() - cx)
    v = (v.to(depth_map.device).float() - cy)

    # Calculate 3D points in camera space
    z = depth_map
    x = (u / fl_px) * z
    y = (v / fl_px) * z
    points_3d = torch.stack((x, y, z), dim=-1)
    return points_3d

def compute_optical_flow(
        img_BCHW: torch.Tensor,
        K: torch.Tensor,
        bTa: torch.Tensor,
        arkit_depth_img_path: Path,
        pda_model,
        data_height: int,
        data_width: int,
    ) -> torch.Tensor:
    # Obtain the ground truth and PromptDA predicted depths
    prompt_depth = cv2.imread(str(arkit_depth_img_path), cv2.IMREAD_UNCHANGED)
    prompt_depth_BCHW = to_tensor_func(prompt_depth / 1000.).float().cuda()
    pda_depth_BCHW = pda_model.predict(img_BCHW, prompt_depth_BCHW)
    depth_map_HW = pda_depth_BCHW.clone().squeeze()

    # Backproject the depths into 3D space
    i_centers = np.arange(data_height)
    j_centers = np.arange(data_width)
    i_idx, j_idx = np.ix_(i_centers, j_centers)
    fx_px = K[0, 0]
    cx, cy = K[0, 2], K[1, 2]
    pts_a_HW3 = backproject(depth_map_HW, fx_px, (cx, cy))
    query_a_N3 = pts_a_HW3[i_idx, j_idx].reshape(-1, 3)
    # Create a 2D grid of query points from the center indices
    j_grid, i_grid = torch.meshgrid(torch.from_numpy(j_centers), torch.from_numpy(i_centers), indexing='xy')
    query_pixels_N2 = torch.stack([j_grid.ravel(), i_grid.ravel()], dim=-1).reshape(-1, 2)

    # Transform the 3D points into the second image's frame
    bRa = bTa[:3, :3].cuda()
    bta = bTa[:3, 3].unsqueeze(-1).cuda()
    aligned_a_N3 = (bRa @ query_a_N3.T + bta).T

    # Project the 3D points into the second image's frame
    projected_a_N3 = (K @ aligned_a_N3.T).T
    projected_a_N2 = projected_a_N3[:, :2] / projected_a_N3[:, 2, None]

    # Compute the optical flow between the two images
    queries_HW2 = query_pixels_N2.reshape(data_height, data_width, 2).cuda()
    projected_HW2 = projected_a_N2.reshape(data_height, data_width, 2)
    flow_2HW = (projected_HW2 - queries_HW2).permute(2,0,1).cpu()

    # Delete intermediate variables to save GPU memory
    del prompt_depth_BCHW, pda_depth_BCHW
    del prompt_depth, i_centers, j_centers, i_idx, j_idx, fx_px, cx, cy
    del pts_a_HW3, query_a_N3, j_grid, i_grid, query_pixels_N2
    del bRa, bta, aligned_a_N3, projected_a_N3, queries_HW2, projected_HW2
    torch.cuda.empty_cache()
    
    return flow_2HW, depth_map_HW

def interpolate_frames(
        img1_BCHW: torch.Tensor,
        img2_BCHW: torch.Tensor,
        img1_frame_num: int,
        rife_model: RIFEModel,
        frame_num_gap: int,
        rgb_path: Path,
        data_height: int,
        data_width: int,
    ) -> torch.Tensor:
    keyframe_imgs = [img1_BCHW]
    for idx in range(1, frame_num_gap):
        next_frame_num = img1_frame_num + idx
        next_filename_jpg = "frame_" + str(next_frame_num).zfill(6) + ".jpg"
        next_img_BCHW = read_img_path(rgb_path / next_filename_jpg, resize=True, data_width=data_width, data_height=data_height)
        keyframe_imgs.append(next_img_BCHW)
    keyframe_imgs.append(img2_BCHW)

    # Stack all frames into a single tensor
    frames_stacked = torch.concat(keyframe_imgs, dim=0)  # Shape [N,C,H,W]

    # Pad the images for RIFE inference
    B, C, H, W = img1_BCHW.shape
    H_p = ((H - 1) // 32 + 1) * 32
    W_p = ((W - 1) // 32 + 1) * 32
    padding = (0, W_p - W, 0, H_p - H)
    frames_padded = F.pad(frames_stacked, padding)

    # Get consecutive pairs for inference
    frames_1 = frames_padded[:-1]  # All frames except last
    frames_2 = frames_padded[1:]   # All frames except first

    # First interpolation pass
    mid_frames = rife_model.inference(frames_1, frames_2).detach()

    # After the first interpolation pass, construct sequence [W, wx, X, xy, Y, yz, Z]
    frames_list = [frames_padded[0]]  # Start with W
    for i in range(len(mid_frames)):
        frames_list.append(mid_frames[i])
        frames_list.append(frames_padded[i+1])

    # Stack for second pass
    curr_frames = torch.stack(frames_list)  # Shape [7,C,H,W]

    # Second interpolation pass - batch process pairs
    batch_frame1 = curr_frames[:-1]  # All frames except last
    batch_frame2 = curr_frames[1:]   # All frames except first
    mid_frames_2 = rife_model.inference(batch_frame1, batch_frame2).detach()

    # Construct final sequence
    final_frames_list = [curr_frames[0]]  # Start with first frame
    for i in range(len(mid_frames_2)):
        final_frames_list.append(mid_frames_2[i])
        final_frames_list.append(curr_frames[i+1])

    final_frames = torch.stack(final_frames_list)  # Shape [15,C,H,W]
    final_frames = final_frames[..., :H, :W]
    
    # Delete intermediate variables to save GPU memory
    del keyframe_imgs, frames_stacked, frames_padded, frames_1, frames_2, mid_frames, frames_list
    del curr_frames, batch_frame1, batch_frame2, mid_frames_2, final_frames_list
    del B, C, H, W, H_p, W_p, padding
    torch.cuda.empty_cache()
    
    return final_frames

VAL_SEQS = [
    "7b6477cb95",
    "c50d2d1d42",
    "cc5237fd77",
    "acd95847c5",
    "fb5a96b1a2",
    "a24f64f7fb",
    "1ada7a0617",
    "5eb31827b7",
    "3e8bba0176",
    "3f15a9266d",
    "21d970d8de",
    "5748ce6f01",
    "c4c04e6d6c",
    "7831862f02",
    "bde1e479ad",
    "38d58a7a31",
    "5ee7c22ba0",
    "f9f95681fd",
    "3864514494",
    "40aec5fffa",
    "13c3e046d7",
    "e398684d27",
    "a8bf42d646",
    "45b0dac5e3",
    "31a2c91c43",
    "e7af285f7d",
    "286b55a2bf",
    "7bc286c1b6",
    "f3685d06a9",
    "b0a08200c9",
    "825d228aec",
    "a980334473",
    "f2dc06b1d2",
    "5942004064",
    "25f3b7a318",
    "bcd2436daf",
    "f3d64c30f8",
    "0d2ee665be",
    "3db0a1c8f3",
    "ac48a9b736",
    "c5439f4607",
    "578511c8a9",
    "d755b3d9d8",
    "99fa5c25e1",
    "09c1414f1b",
    "5f99900f09",
    "9071e139d9",
    "6115eddb86",
    "27dd4da69e",
    "c49a8c6cff"
]
