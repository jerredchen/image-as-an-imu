import torch
import numpy as np

# from PoseDiffusion:
# https://github.com/facebookresearch/PoseDiffusion/blob/main/pose_diffusion/util/metric.py
def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / torch.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg

def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=-1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=-1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def find_closest_timestamps_indices(frame_ts, sensor_ts):
    """Find closest timestamp indices for alignment."""
    frame_ts = np.asarray(frame_ts)
    sensor_ts = np.asarray(sensor_ts)
    
    idx = np.searchsorted(sensor_ts, frame_ts)
    idx = np.clip(idx, 1, len(sensor_ts) - 1)
    
    left_distances = np.abs(sensor_ts[idx-1] - frame_ts)
    right_distances = np.abs(sensor_ts[idx] - frame_ts)
    
    closest_idx = idx - (left_distances < right_distances)
    
    return closest_idx

def compute_rmse(pred_vels_N3, gt_vels_N3, pred_ts, gt_ts):
    closest_indices = find_closest_timestamps_indices(pred_ts, gt_ts)
    abs_error = np.abs(pred_vels_N3 - gt_vels_N3[closest_indices])

    rmse = np.sqrt(np.mean(abs_error**2, axis=0))

    return rmse