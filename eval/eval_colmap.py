import time
import torch
import roma
import hydra
import numpy as np
import pycolmap
from pathlib import Path
from omegaconf import DictConfig
from hloc import extract_features, match_features, pairs_from_retrieval, reconstruction
from evo.core.trajectory import PoseTrajectory3D
from iaai.utils.viz import viz_time_series
from iaai.utils.io import load_strayscanner_data
from iaai.utils.metrics import compute_rmse

@hydra.main(config_path="./configs", config_name="colmap_cfg", version_base="1.3")
def main(cfg: DictConfig):
    """Evaluate COLMAP baseline."""
    seq_dir = Path(cfg.data_path)
    
    # Load data
    data = load_strayscanner_data(seq_dir)
    rgb_files = data['rgb_files']
    frame_ts = data['frame_ts']
    imu_ts = data['imu_ts']
    arkit_ts = data['arkit_ts']
    gt_rots = data['gt_rots']
    gt_trans = data['gt_trans']
    imu_data = data['imu_data']
    vel_data = data['vel_data']
    
    # Setup COLMAP directories
    if cfg.feature_conf == "disk" and cfg.matcher_conf == "disk+lightglue":
        colmap_outputs_dir = seq_dir / "colmap_outputs_dlg"
    else:
        colmap_outputs_dir = seq_dir / "colmap_outputs"
    
    # Setup output directory
    suffix = "" if cfg.save_suffix is None else "_" + cfg.save_suffix
    precomputed_vels_dir = seq_dir / f"precomputed_vels_colmap{suffix}"
    precomputed_vels_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if precomputed velocities exist
    if cfg.precomputed_vels_path:
        pred_trans_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_trans_vels.pt")
        pred_rot_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_rot_vels.pt")
        selected_frame_indices = torch.load(Path(cfg.precomputed_vels_path) / "selected_frame_indices.pt")
    else:
        # Setup COLMAP directories and files
        use_existing_colmap_outputs = False
        if colmap_outputs_dir.exists() and not cfg.overwrite_colmap_outputs:
            print("Reusing existing colmap_outputs directory.")
            use_existing_colmap_outputs = True
        elif colmap_outputs_dir.exists() and cfg.overwrite_colmap_outputs:
            print("Overwriting existing colmap_outputs directory.")
            import shutil
            shutil.rmtree(colmap_outputs_dir)
            colmap_outputs_dir.mkdir(parents=True, exist_ok=False)
        else:
            colmap_outputs_dir.mkdir(parents=True, exist_ok=False)
        
        sfm_pairs = colmap_outputs_dir / 'pairs-sfm.txt'
        sfm_dir = colmap_outputs_dir / 'sfm'
        features = colmap_outputs_dir / 'features.h5'
        matches = colmap_outputs_dir / 'matches.h5'
        
        retrieval_conf = extract_features.confs[cfg.retrieval_conf]
        feature_conf = extract_features.confs[cfg.feature_conf]
        matcher_conf = match_features.confs[cfg.matcher_conf]
        
        rgb_dir_path = seq_dir / "rgb_320x224"
        
        # Extract features and matches
        if not use_existing_colmap_outputs:
            references = [str(rgb_file.relative_to(rgb_dir_path)) for rgb_file in rgb_files]
            retrieval_path = extract_features.main(retrieval_conf, rgb_dir_path, colmap_outputs_dir)
            pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
            start_time = time.time()
            extract_features.main(feature_conf, rgb_dir_path, image_list=references, feature_path=features)
            match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
            reconstruction.main(sfm_dir, rgb_dir_path, sfm_pairs, features, matches, image_list=references)
            end_time = time.time()
            print(f"Time taken for feature extraction, matching and reconstruction: {end_time - start_time} seconds")

        # Load all reconstruction models and map image IDs to models
        reconstructions = []
        img_model_map = {}
        model_paths = list(sfm_dir.glob("**/cameras.bin"))
        for i, model_path in enumerate(model_paths):
            model = pycolmap.Reconstruction(model_path.parent)
            reconstructions.append(model)
            for img_id in model.images.keys():
                img_model_map.setdefault(img_id, []).append(i)
            print(f"Model {len(reconstructions)}: {len(model.images)} images")
        print(f"\nTotal unique images across all models: {len(img_model_map)}")

        # Select frames and collect colmap poses (world_from_cam)
        sorted_img_ids = sorted(img_model_map.keys())
        selected_frame_indices = []
        colmap_poses = []
        selected_ts = []
        for k, img_id in enumerate(sorted_img_ids):
            model_id = img_model_map[img_id][0]
            model = reconstructions[model_id]
            pose = model.images[img_id].cam_from_world.inverse().matrix()
            selected_frame_indices.append(img_id - 1)
            colmap_poses.append(pose)
            selected_ts.append(frame_ts[img_id - 1])
        
        torch.save(selected_frame_indices, precomputed_vels_dir / 'selected_frame_indices.pt')

        # Alignment of COLMAP poses to ground truth
        colmap_positions = []
        colmap_quats = []
        for pose in colmap_poses:
            pos = pose[:3, 3]
            R = pose[:3, :3]
            quat_XYZW = roma.rotmat_to_unitquat(torch.from_numpy(R))
            colmap_positions.append(pos)
            colmap_quats.append(torch.hstack((quat_XYZW[-1][None], quat_XYZW[:-1])))
        colmap_positions = np.stack(colmap_positions)
        colmap_quats = np.stack(colmap_quats)

        # Create trajectory for COLMAP poses
        colmap_traj = PoseTrajectory3D(
            positions_xyz=colmap_positions,
            orientations_quat_wxyz=colmap_quats,
            timestamps=np.array(selected_ts)
        )

        # Build ground-truth trajectory for the same frames
        gt_positions = []
        gt_quats = []
        for img_ind in selected_frame_indices:
            gt_positions.append(gt_trans[img_ind])
            gt_quats.append(gt_rots[img_ind])
        gt_positions = np.stack(gt_positions)
        gt_quats = np.stack(gt_quats)
        gt_traj_selected = PoseTrajectory3D(
            positions_xyz=gt_positions,
            orientations_quat_wxyz=gt_quats,
            timestamps=np.array(selected_ts)
        )

        # Align the COLMAP trajectory to ground truth
        colmap_traj.align(gt_traj_selected, correct_scale=True, correct_only_scale=False)
        aligned_poses = colmap_traj.poses_se3

        # Compute velocities from aligned poses
        pred_trans_vels = []
        pred_rot_vels = []
        for i in range(1, len(aligned_poses)-1):
            dt = selected_ts[i+1] - selected_ts[i-1]
            # Translational velocity
            t1 = aligned_poses[i-1][:3, 3]
            t2 = aligned_poses[i+1][:3, 3]
            trans_vel = (t2 - t1) / dt
            pred_trans_vels.append(torch.tensor(trans_vel, dtype=torch.float))
            
            # Rotational velocity
            wRa = aligned_poses[i-1][:3, :3]
            wRb = aligned_poses[i+1][:3, :3]
            M = np.array([[0, -1, 0],
                          [-1, 0, 0],
                          [0, 0, -1]], dtype=np.float64)
            aRb = wRa.T @ wRb
            aRb = M @ aRb @ M.T
            euler_diff = roma.rotmat_to_euler('XYZ', torch.from_numpy(aRb))
            rot_vel = euler_diff / dt
            pred_rot_vels.append(rot_vel)
        
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
    
    frame_ts_selected = frame_ts[selected_frame_indices][1:-1]
    rot_rmse = compute_rmse(pred_rot_vels, imu_gyro_N3, frame_ts_selected, imu_ts)
    trans_rmse = compute_rmse(pred_trans_vels, arkit_tvel_N3, frame_ts_selected, arkit_ts)
    print(f"Rotational RMSE: {rot_rmse[0]:.2f} / {rot_rmse[1]:.2f} / {rot_rmse[2]:.2f}")
    print(f"Translational RMSE: {trans_rmse[0]:.2f} / {trans_rmse[1]:.2f} / {trans_rmse[2]:.2f}")
    
    if cfg.viz_time_series:
        title_prefix = "COLMAP"
        rot_title = title_prefix + " - Angular Vels vs Gyroscope"
        trans_title = title_prefix + " - Translational Vels vs ARKit"
        
        viz_time_series(pred_rot_vels, imu_gyro_N3, frame_ts_selected, imu_ts, 
                       rot_or_trans="rot", title=rot_title)
        viz_time_series(pred_trans_vels, arkit_tvel_N3, frame_ts_selected, arkit_ts, 
                       rot_or_trans="trans", title=trans_title)

if __name__ == "__main__":
    main() 