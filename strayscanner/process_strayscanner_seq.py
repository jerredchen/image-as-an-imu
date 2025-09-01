import hydra
from pathlib import Path
from omegaconf import OmegaConf

from process_strayscanner_utils import *

@hydra.main(config_path="./configs", config_name="process_strayscanner_cfg", version_base="1.2")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    data_path = Path(cfg.DATA_PATH)
    assert data_path.exists(), f"Data path {data_path} does not exist"

    # Load and process pose data
    arkit_poses, odom_data, imu_data = load_and_process_poses(data_path)

    # Apply pose convention transformation if needed
    opencv_poses = arkit_to_opencv(arkit_poses)

    # Plot poses and/or IMU if enabled
    plot_poses_and_imu(opencv_poses, imu_data, cfg.PLOT_POSES, cfg.PLOT_IMU)

    if cfg.EXTRACT_FRAMES:
        extract_frames_from_video(data_path)
    process_images(data_path, cfg.RESIZE, cfg.CROP, cfg.RGB_SUFFIX)

    # Compute trajectory and velocities
    trajectory_aTbs, trans_vels, rot_vels, timestamps = compute_trajectory_velocities(opencv_poses, odom_data)

    # Detect and save blurry frames if enabled
    detect_and_save_blurry_frames(data_path, trajectory_aTbs, timestamps, odom_data, cfg.BLURRY_DETECT)

    # Align IMU data with pose timestamps
    closest_idx = find_closest_timestamps_indices(odom_data['timestamp'].values[1:], imu_data['timestamp'].values)
    imu_aligned_ts = imu_data['timestamp'].values[closest_idx]
    imu_aligned_ts = imu_aligned_ts - imu_aligned_ts[0]

    # Plot velocities if enabled
    if cfg.PLOT_VELS:
        plot_velocities(trans_vels, timestamps, imu_data, closest_idx, imu_aligned_ts)

    # Save final outputs
    save_final_outputs(data_path, arkit_poses, opencv_poses, odom_data, trans_vels, imu_data, closest_idx)

    print(f"{colors.OKGREEN}Done!{colors.ENDC}")

if __name__ == "__main__":
    main()