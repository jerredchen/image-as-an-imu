import torch
import roma
import hydra
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from evo.core.trajectory import PoseTrajectory3D
from iaai.utils.viz import viz_time_series
from iaai.utils.io import load_strayscanner_data
from iaai.utils.metrics import compute_rmse

@hydra.main(config_path="./configs", config_name="droid_cfg", version_base="1.3")
def main(cfg: DictConfig):
    """Evaluate DROID-SLAM baseline."""
    seq_dir = Path(cfg.data_path)
    seq_name = seq_dir.stem
    
    # Load DROID poses
    droid_poses = np.load(Path(cfg.droid_output_dir) / f"{seq_name}" / "poses.npy")
    
    # Load data
    data = load_strayscanner_data(seq_dir)
    frame_ts = data['frame_ts']
    imu_ts = data['imu_ts']
    arkit_ts = data['arkit_ts']
    gt_rots = data['gt_rots']
    gt_trans = data['gt_trans']
    imu_data = data['imu_data']
    vel_data = data['vel_data']
    
    # Setup output directory
    suffix = "" if cfg.save_suffix is None else "_" + cfg.save_suffix
    precomputed_vels_dir = seq_dir / f"precomputed_vels_droid{suffix}"
    precomputed_vels_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if precomputed velocities exist
    if cfg.precomputed_vels_path:
        pred_trans_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_trans_vels.pt")
        pred_rot_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_rot_vels.pt")
    else:
        # Align DROID poses to ground truth
        droid_traj = PoseTrajectory3D(
            positions_xyz=droid_poses[:, :3],
            orientations_quat_wxyz=np.hstack((droid_poses[:,-1][:,None], droid_poses[:,3:])),
            timestamps=frame_ts
        )
        gt_traj = PoseTrajectory3D(
            positions_xyz=gt_trans,
            orientations_quat_wxyz=gt_rots,
            timestamps=frame_ts
        )

        droid_traj.align(gt_traj, correct_scale=True, correct_only_scale=False)
        droid_poses = droid_traj.poses_se3

        # Compute velocities from aligned poses
        pred_trans_vels = []
        pred_rot_vels = []
        
        for k in range(1, len(droid_poses)-1):
            wTa = droid_poses[k-1]
            wTb = droid_poses[k+1]
            
            M = torch.tensor([[0, -1, 0],
                              [-1, 0, 0],
                              [0, 0, -1]]).float()
            
            aRb_pred = M @ torch.from_numpy(wTa[:3, :3].T @ wTb[:3, :3]).float() @ M.T
            atb_pred = torch.from_numpy(wTa[:3, :3].T @ (wTa[:3, 3] - wTb[:3, 3])[:,None]).float().squeeze()
            
            pred_trans_vels.append(atb_pred / (frame_ts[k+1] - frame_ts[k-1]))
            pred_rot_vels.append(-roma.rotmat_to_euler('XYZ', aRb_pred) / (frame_ts[k+1] - frame_ts[k-1]))
        
        pred_trans_vels = torch.stack(pred_trans_vels)
        pred_rot_vels = torch.stack(pred_rot_vels)
        
        if cfg.save_vels:
            torch.save(pred_trans_vels, precomputed_vels_dir / "pred_trans_vels.pt")
            torch.save(pred_rot_vels, precomputed_vels_dir / "pred_rot_vels.pt")
    
    # Evaluate against ground truth
    imu_gyro_N3 = imu_data[['alpha_x', 'alpha_y', 'alpha_z']].values
    arkit_tvel_N3 = vel_data[['t_x', 't_y', 't_z']].values
    
    pred_trans_vels = pred_trans_vels.cpu().numpy()
    pred_rot_vels = pred_rot_vels.cpu().numpy()

    # First and last frame are unused for DROID
    frame_ts = frame_ts[1:-1]
    rot_rmse = compute_rmse(pred_rot_vels, imu_gyro_N3, frame_ts, imu_ts)
    trans_rmse = compute_rmse(pred_trans_vels, arkit_tvel_N3, frame_ts, arkit_ts)
    print(f"Rotational RMSE: {rot_rmse[0]:.2f} / {rot_rmse[1]:.2f} / {rot_rmse[2]:.2f}")
    print(f"Translational RMSE: {trans_rmse[0]:.2f} / {trans_rmse[1]:.2f} / {trans_rmse[2]:.2f}")
    
    if cfg.viz_time_series:
        title_prefix = "DROID-SLAM"
        rot_title = title_prefix + " - Angular Vels vs Gyroscope"
        trans_title = title_prefix + " - Translational Vels vs ARKit"
        
        viz_time_series(pred_rot_vels, imu_gyro_N3, frame_ts, imu_ts, 
                       rot_or_trans="rot", title=rot_title)
        viz_time_series(pred_trans_vels, arkit_tvel_N3, frame_ts, arkit_ts, 
                       rot_or_trans="trans", title=trans_title)

if __name__ == "__main__":
    main() 