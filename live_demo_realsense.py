import argparse
import time
from collections import deque

import torch
import roma
import visdom
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from torchvision.transforms.functional import normalize, resize, center_crop

from iaai.model import Blur2PoseSegNeXtBackbone
from iaai.funcs import disambiguate_direction
from iaai.utils.io import load_ckpt

# Initialize Visdom
viz = visdom.Visdom()

# Configure RealSense pipeline
pipe = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(config)

img_buffer = deque(maxlen=3)
frame_ts = deque(maxlen=3)

# Data storage for velocity plotting (store 10 seconds of data at 30 FPS = ~300 samples)
max_samples = 300
vel_timestamps = deque(maxlen=max_samples)
rot_vel_x = deque(maxlen=max_samples)
rot_vel_y = deque(maxlen=max_samples)
rot_vel_z = deque(maxlen=max_samples)
trans_vel_x = deque(maxlen=max_samples)
trans_vel_y = deque(maxlen=max_samples)
trans_vel_z = deque(maxlen=max_samples)

def update_velocity_plot():
    """Create and update the 6-subplot velocity plot"""
    if len(vel_timestamps) < 2:
        return
        
    # Convert timestamps to relative time (seconds from start)
    times = np.array(vel_timestamps)
    times = (times - times[0]) / 1000.0  # Convert to seconds from start
    
    # Create the plot
    fig, axes = plt.subplots(6, 1, figsize=(10, 6))
    fig.suptitle('Rotational and Translational Velocities', fontsize=14)
    
    # Plot rotational velocities
    axes[0].plot(times, rot_vel_x, 'r-', linewidth=2)
    axes[0].set_ylabel('Rot Vel X (rad/s)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(times, rot_vel_y, 'g-', linewidth=2)
    axes[1].set_ylabel('Rot Vel Y (rad/s)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(times, rot_vel_z, 'b-', linewidth=2)
    axes[2].set_ylabel('Rot Vel Z (rad/s)')
    axes[2].grid(True, alpha=0.3)
    
    # Plot translational velocities
    axes[3].plot(times, trans_vel_x, 'r--', linewidth=2)
    axes[3].set_ylabel('Trans Vel X (m/s)')
    axes[3].grid(True, alpha=0.3)
    
    axes[4].plot(times, trans_vel_y, 'g--', linewidth=2)
    axes[4].set_ylabel('Trans Vel Y (m/s)')
    axes[4].grid(True, alpha=0.3)
    
    axes[5].plot(times, trans_vel_z, 'b--', linewidth=2)
    axes[5].set_ylabel('Trans Vel Z (m/s)')
    axes[5].set_xlabel('Time (seconds)')
    axes[5].grid(True, alpha=0.3)
    
    # Set x-axis limits to show only last 10 seconds
    if times[-1] > 10:
        for ax in axes:
            ax.set_xlim(times[-1] - 10, times[-1])
    
    plt.tight_layout()
    return fig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth or .safetensors)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--exposure", type=float, default=0.01, help="Exposure time in seconds (used to scale velocity)")
    p.add_argument("--fl", type=float, default=500.0, help="Focal length in pixels")
    return p.parse_args()


args = parse_args()

# Load model
device = args.device
model = Blur2PoseSegNeXtBackbone(device=device)
checkpoint = load_ckpt(args.ckpt)
model.load_state_dict(checkpoint, strict=False)
model.eval()
model.to(device)

opencv2gyro_tf = torch.tensor([[ 0, -1,  0],
                               [-1,  0,  0],
                               [ 0,  0, -1]]).float().to(device)
try:
    while True:
        frames = pipe.wait_for_frames()

        # --- Handle Color Frame ---
        color_frame = frames.get_color_frame()
        if color_frame:
            timestamp = frames.get_timestamp()
            color_image_rgb = np.asanyarray(color_frame.get_data())
            viz.image(
                color_image_rgb.transpose(2, 0, 1),
                win='color_feed',
                opts=dict(title="Color Feed")
            )
            rgb_img = torch.from_numpy(color_image_rgb).float() / 255.0
            rgb_img = rgb_img.unsqueeze(0).permute(0, 3, 1, 2).to(device)
            rgb_img = resize(rgb_img, size=(252, 336), antialias=True)
            rgb_img = center_crop(rgb_img, (224, 320))
            img_buffer.append(rgb_img)
            frame_ts.append(timestamp)
            if len(img_buffer) == 3:
                norm_img = normalize(img_buffer[1], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                data = {"image": norm_img, "fl": torch.tensor([args.fl]).float().to(device)}
                with torch.no_grad():
                    out = model(data)
                pred_pose_B6 = out["pose"].squeeze()
                flow_HW2 = out["flow_field"].squeeze().permute(1, 2, 0)
                depth_HW = out["depth"].squeeze().cpu()
                aRb_pred = roma.euler_to_rotmat('XYZ', pred_pose_B6[:, :3]).squeeze().cpu()
                atb_pred = pred_pose_B6[:, 3:].squeeze().cpu()

                rot_vel, trans_vel = disambiguate_direction(
                    flow_HW2.cpu(),
                    args.exposure,
                    frame_ts[1] - frame_ts[0],
                    aRb_pred.float(),
                    atb_pred.float(),
                    img_buffer[1].squeeze(0).cpu(),
                    img_buffer[2].squeeze(0).cpu(),
                    img_buffer[0].squeeze(0).cpu() if len(img_buffer) > 0 else None,
                    opencv2gyro_tf
                )
                # Store velocity data for plotting
                vel_timestamps.append(timestamp)
                # Convert to float in case velocities are tensors
                rot_vel_x.append(rot_vel[0].item())
                rot_vel_y.append(rot_vel[1].item())
                rot_vel_z.append(rot_vel[2].item())
                trans_vel_x.append(trans_vel[0].item())
                trans_vel_y.append(trans_vel[1].item())
                trans_vel_z.append(trans_vel[2].item())

                # Update the plot every few frames to avoid overloading
                if len(vel_timestamps) >= 2 and len(vel_timestamps) % 1 == 0:
                    fig = update_velocity_plot()
                    if fig is not None:
                        viz.matplot(fig, win='velocity')
                        plt.close(fig)


        time.sleep(0.01)  # Small sleep to avoid overloading CPU

finally:
    pipe.stop()