import torch
import roma
import hydra
import cv2
from pathlib import Path
from omegaconf import DictConfig
from torchvision.transforms.functional import normalize
from iaai.model import Blur2PoseSegNeXtBackbone
from iaai.funcs import disambiguate_direction
from iaai.utils.viz import viz_time_series
from iaai.utils.io import load_strayscanner_data, load_ckpt, read_img_path
from iaai.utils.metrics import compute_rmse

@hydra.main(config_path="./configs", config_name="iaai_cfg", version_base="1.3")
def main(cfg: DictConfig):
    """Evaluate blur model."""
    seq_dir = Path(cfg.data_path)
    
    # Load model
    checkpoint = load_ckpt(cfg.ckpt_path)
    model = Blur2PoseSegNeXtBackbone(supervise_pose=True)
    model.load_state_dict(checkpoint, strict=False)
    model.eval().cuda()
    
    # Load data
    data = load_strayscanner_data(seq_dir)
    rgb_files = data['rgb_files']
    frame_ts = data['frame_ts']
    imu_ts = data['imu_ts']
    arkit_ts = data['arkit_ts']
    exposure_time_s = data['exposure_time_s']
    fx, fy = data['fx'], data['fy']
    imu_data = data['imu_data']
    vel_data = data['vel_data']
    
    # Setup output directory
    suffix = "" if cfg.save_suffix is None else "_" + cfg.save_suffix
    precomputed_vels_dir = seq_dir / f"precomputed_vels_iaai{suffix}"
    precomputed_vels_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if precomputed velocities exist
    if cfg.precomputed_vels_path:
        pred_trans_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_trans_vels.pt")
        pred_rot_vels = torch.load(Path(cfg.precomputed_vels_path) / "pred_rot_vels.pt")
    else:
        # Run inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        pred_trans_vels = []
        pred_rot_vels = []
        aRb_preds = []
        atb_preds = []
        opencv2gyro_tf = torch.tensor([[ 0, -1,  0],
                                       [-1,  0,  0],
                                       [ 0,  0, -1]]).float()
        for k, rgb_file in enumerate(rgb_files):
            rgb_img = read_img_path(rgb_file)
            assert rgb_img.shape[-3:] == (3, 224, 320)
            rgb_norm = normalize(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            with torch.no_grad():
                data = {
                    "image": rgb_norm,
                    "fl": torch.tensor([0.5*(fx+fy)]).float()
                }
                out = model(data)
            flow_HW2 = out["flow_field"].squeeze().permute(1, 2, 0)
            depth_HW = out["depth"].squeeze()
            pred_pose_B6 = out["pose"].squeeze(-1)
            residual = out["residual"].squeeze()
            aRb_pred = roma.euler_to_rotmat('XYZ', pred_pose_B6[:, :3]).squeeze().cpu()
            atb_pred = pred_pose_B6[:, 3:].squeeze().cpu()

            # Disambiguate motion direction
            if k < len(rgb_files) - 1:
                rot_vel, trans_vel = disambiguate_direction(
                    flow_HW2.cpu(),
                    exposure_time_s[k],
                    (frame_ts[k+1] - frame_ts[k]),
                    aRb_pred.float(),
                    atb_pred.float(),
                    rgb_img.squeeze(0).cpu(),
                    read_img_path(rgb_files[k+1], device="cpu").squeeze(0),
                    read_img_path(rgb_files[k-1], device="cpu").squeeze(0) if k > 0 else None,
                    opencv2gyro_tf
                )
                if cfg.oracle_direction:
                    trans_vel = torch.abs(trans_vel) * torch.sign(torch.tensor(vel_data[['t_x', 't_y', 't_z']].values[k]))
                    rot_vel = torch.abs(rot_vel) * torch.sign(torch.tensor(vel_data[['alpha_x', 'alpha_y', 'alpha_z']].values[k]))
                pred_trans_vels.append(trans_vel)
                pred_rot_vels.append(rot_vel)
                aRb_preds.append(roma.euler_to_rotmat('XYZ', rot_vel * (frame_ts[k+1] - frame_ts[k])))
                atb_preds.append(trans_vel * (frame_ts[k+1] - frame_ts[k]))
        
        end.record()
        torch.cuda.synchronize()
        print(f"Time taken for {len(rgb_files)} frames, {frame_ts[k] - frame_ts[0]} s video: {start.elapsed_time(end)/1000} s")
        
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

    # Last frame is unused for blur model
    frame_ts = frame_ts[:-1]
    rmse_rot = compute_rmse(pred_rot_vels, imu_gyro_N3, frame_ts, imu_ts)
    rmse_trans = compute_rmse(pred_trans_vels, arkit_tvel_N3, frame_ts, arkit_ts)
    print(f"Rotational RMSE: {rmse_rot[0]:.2f} / {rmse_rot[1]:.2f} / {rmse_rot[2]:.2f}")
    print(f"Translational RMSE: {rmse_trans[0]:.2f} / {rmse_trans[1]:.2f} / {rmse_trans[2]:.2f}")
    
    if cfg.viz_time_series:
        title_prefix = "IAAI"
        rot_title = title_prefix + " - Angular Vels vs Gyroscope"
        trans_title = title_prefix + " - Translational Vels vs ARKit"
        
        viz_time_series(pred_rot_vels, imu_gyro_N3, frame_ts, imu_ts, 
                       rot_or_trans="rot", title=rot_title)
        viz_time_series(pred_trans_vels, arkit_tvel_N3, frame_ts, arkit_ts, 
                       rot_or_trans="trans", title=trans_title)

if __name__ == "__main__":
    main() 