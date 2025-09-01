import os
import sys
import torch
import roma
import hydra
from pathlib import Path
from omegaconf import DictConfig
from iaai.utils.viz import viz_time_series
from iaai.utils.io import load_strayscanner_data
from iaai.utils.metrics import compute_rmse

# MASt3R imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mast3r')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../mast3r/dust3r')))
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

@hydra.main(config_path="./configs", config_name="mast3r_cfg", version_base="1.3")
def main(cfg: DictConfig):
    """Evaluate MASt3R baseline."""
    seq_dir = Path(cfg.data_path)
    
    # Load model
    model = AsymmetricMASt3R.from_pretrained(cfg.mast3r_ckpt_name).cuda()
    model.eval().cuda()
    
    # Load data
    data = load_strayscanner_data(seq_dir)
    rgb_files = data['rgb_files']
    frame_ts = data['frame_ts']
    imu_ts = data['imu_ts']
    arkit_ts = data['arkit_ts']
    imu_data = data['imu_data']
    vel_data = data['vel_data']
    
    # Setup output directory
    suffix = "" if cfg.save_suffix is None else "_" + cfg.save_suffix
    precomputed_vels_dir = seq_dir / f"precomputed_vels_mast3r{suffix}"
    precomputed_vels_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if precomputed velocities exist
    if cfg.precomputed_vels_path:
        pred_trans_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_trans_vels.pt")
        pred_rot_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_rot_vels.pt")
    else:
        # Run inference
        pred_trans_vels = []
        pred_rot_vels = []
        
        for k in range(len(rgb_files)):
            if k == 0 or k == len(rgb_files) - 1:
                print(f"Skipping first/last frame for MASt3R")
                continue
            
            rgb_file1, rgb_file2 = rgb_files[k-1], rgb_files[k+1]
            
            # Prepare next and previous images for MASt3R
            images = load_images([str(rgb_file1), str(rgb_file2)], size=512, verbose=False)
            
            # Create image pairs and run inference
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, model, 'cuda', batch_size=1, verbose=False)
            
            # Use the global aligner in PairViewer mode to get poses
            scene = global_aligner(output, device='cuda', mode=GlobalAlignerMode.PairViewer)
            
            poses = scene.get_im_poses()
            wTa, wTb = poses[0], poses[1]
            aTb = (wTa.inverse() @ wTb).cpu()
            
            M = torch.tensor([[ 0, -1,  0],
                              [-1,  0,  0],
                              [ 0,  0, -1]]).float()
            aRb_pred = M @ aTb[:3, :3] @ M.T
            atb_pred = aTb[:3, 3]
            
            pred_trans_vels.append(atb_pred / (frame_ts[k+1] - frame_ts[k-1]))
            pred_rot_vels.append(roma.rotmat_to_euler('XYZ', aRb_pred) / (frame_ts[k+1] - frame_ts[k-1]))
            
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

    # First and last frame are unused for MASt3R
    frame_ts = frame_ts[1:-1]
    rot_rmse = compute_rmse(pred_rot_vels, imu_gyro_N3, frame_ts, imu_ts)
    trans_rmse = compute_rmse(pred_trans_vels, arkit_tvel_N3, frame_ts, arkit_ts)
    print(f"Rotational RMSE: {rot_rmse[0]:.2f} / {rot_rmse[1]:.2f} / {rot_rmse[2]:.2f}")
    print(f"Translational RMSE: {trans_rmse[0]:.2f} / {trans_rmse[1]:.2f} / {trans_rmse[2]:.2f}")
    
    if cfg.viz_time_series:
        title_prefix = "MASt3R"
        rot_title = title_prefix + " - Angular Vels vs Gyroscope"
        trans_title = title_prefix + " - Translational Vels vs ARKit"
        
        viz_time_series(pred_rot_vels, imu_gyro_N3, frame_ts, imu_ts, 
                       rot_or_trans="rot", title=rot_title)
        viz_time_series(pred_trans_vels, arkit_tvel_N3, frame_ts, arkit_ts, 
                       rot_or_trans="trans", title=trans_title)

if __name__ == "__main__":
    main() 