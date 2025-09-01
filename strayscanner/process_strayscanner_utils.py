import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2
from scipy.spatial.transform import Rotation as R, Slerp
import os
import shutil
import pandas as pd
import plotly.subplots as sp
from PIL import Image
from pathlib import Path
from torchvision.io import read_image, write_jpeg
from torchvision.transforms import Resize, CenterCrop

# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def create_coord_frames(poses, sample=1, scale=1, width=5, opacity=1.0, colors=None, legendgroup=None):
    traces = []
    for pose_i in range(0, len(poses), sample):
        # pose is a 4x4 matrix
        pose = poses[pose_i]
        # Extract the rotation and translation components
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        x0, y0, z0 = translation[0], translation[1], translation[2]
        # Iterate three times to create an x, y, and z vector respectively
        if colors is None:
            colors = ['red', 'green', 'blue']
        for i in range(3):
            trace = go.Scatter3d(
                x=[x0, (x0 + scale*rotation[0, i])],
                y=[y0, (y0 + scale*rotation[1, i])],
                z=[z0, (z0 + scale*rotation[2, i])],
                mode='lines',
                line=dict(color=colors[i], width=width),
                hoverinfo='skip',
                showlegend=False,
                opacity=opacity,
                legendgroup=legendgroup
            )
            traces.append(trace)
    return traces

def plot_SE3(poses,
             title='',
             legend_name='',
             fig=None,
             show_plot=True,
             symbol=None,
             save_path="",
             show_frame_num=False,
             show_sec=False,
             legendgroup=None,
             rowcol=(None, None),
             color=None,
             params={}):
    """
    Plot the SE(3) poses as 3D lines or points.
    Parameters:
    - poses: numpy array of shape (N, 4, 4), where N is the number of poses
    - title: str, title of the plot
    - legend_name: str, name of the legend
    - fig: plotly.graph_objects.Figure, plotly figure object
    - show_plot: bool, whether to show the plot immediately
    - symbol: str, type of symbol to plot, either 'points' or 'frames'
    - save_path: str, path to save the plot as an HTML file
    - show_frame_num: bool, whether to show frame numbers

    """

    # Obtain the poses as a numpy array
    frame_num_range = params["frame_num_range"] if "frame_num_range" in params else [len(poses)]
    # Extract the translations from the SE(3) 4x4 matrices
    translations = poses[:, :3, 3]
    if fig is None:
        fig = go.Figure()
    if not symbol:
        fig.add_trace(
            go.Scatter3d(
                x=translations[:,0],
                y=translations[:,1],
                z=translations[:,2],
                mode='lines',
                # line=dict(color=color, width=2)
                line=dict(width=7, color=np.linspace(0, 1, len(poses)), colorscale='viridis', colorbar=dict(title='Trajectory Location'))
            ),
            row=rowcol[0],
            col=rowcol[1]
        )
    elif symbol == "points":
        # Create a 3D line plot
        fig.add_trace(
            go.Scatter3d(
                x=translations[:,0],
                y=translations[:,1],
                z=translations[:,2],
                marker=dict(
                    color=color,
                    size=2
                ),
                legendgroup=legendgroup,
                line=dict(width=7, color=np.linspace(0, 1, len(poses)), colorscale='viridis', colorbar=dict(title='Trajectory Location'))
            ),
            row=rowcol[0],
            col=rowcol[1]
        )
        # Set trace name for legend
        fig.data[-1].name = legend_name
    elif symbol == "frames":
        if show_frame_num:
            text = [f"{i}" for i in range(*frame_num_range)]
        elif show_sec:
            text = [f"{i/30:.2f} s" for i in range(*frame_num_range)]
        else:
            text = None
        fig.add_trace(go.Scatter3d(
            x=translations[:,0],
            y=translations[:,1],
            z=translations[:,2],
            mode='lines+text' if (show_frame_num or show_sec) else 'lines',
            text=text,
            # line=dict(width=10, dash='dot', color=color),
            legendgroup=legendgroup,
            line=dict(width=7, color=np.linspace(0, 1, len(poses)), colorscale='viridis', colorbar=dict(title='Trajectory Location'))
        ),
        row=rowcol[0],
        col=rowcol[1]
        )
        # Set trace name for legend
        fig.data[-1].name = legend_name
        sample = params["sample"] if "sample" in params else 1
        scale = params["scale"] if "scale" in params else 1
        width = params["width"] if "width" in params else 5
        opacity = params["opacity"] if "opacity" in params else 1.0
        colors = params["colors"] if "colors" in params else None
        traces = create_coord_frames(
            poses,
            sample=sample,
            scale=scale,
            width=width,
            opacity=opacity,
            colors=colors,
            legendgroup=legendgroup
        )

        fig.add_traces(traces)

    else:
        raise AssertionError("Not valid symbol")

    # Set plot layout
    camera = dict(
        up=dict(x=0, y=1, z=0),
        eye=dict(x=0.01, y=0, z=0),
        center=dict(x=0, y=0, z=0)
    )
    fig.update_layout(scene=dict(aspectmode='data', camera=camera), title=title)

    if show_plot:
        # Show the plot
        fig.show()

    if len(save_path):
        fig.write_html(save_path)

    return fig

def interpolate_pose(pose_prev, pose_curr, alpha):
    """
    Interpolate between two SE(3) poses.
    
    Parameters:
        pose_prev: (4, 4) SE(3) matrix for the previous pose
        pose_curr: (4, 4) SE(3) matrix for the current pose
        alpha: interpolation factor (0.0 -> pose_prev, 1.0 -> pose_curr)
        
    Returns:
        interpolated_pose: (4, 4) SE(3) matrix
    """
    # Decompose poses into rotation (quaternion) and translation
    rot_prev = R.from_matrix(pose_prev[:3, :3])
    rot_curr = R.from_matrix(pose_curr[:3, :3])
    trans_prev = pose_prev[:3, 3]
    trans_curr = pose_curr[:3, 3]
    
    # Slerp for rotation
    key_times = [0, 1]  # Define the range of interpolation (0: pose_prev, 1: pose_curr)
    slerp = Slerp(key_times, R.concatenate([rot_prev, rot_curr]))
    interpolated_rot = slerp([alpha]).as_matrix()[0]
    
    # Linear interpolation for translation
    interpolated_trans = (1 - alpha) * trans_prev + alpha * trans_curr
    
    # Reassemble the SE(3) pose
    interpolated_pose = np.eye(4)
    interpolated_pose[:3, :3] = interpolated_rot
    interpolated_pose[:3, 3] = interpolated_trans
    
    return interpolated_pose

def calculate_relative_pose(pose_a, pose_b):
    """
    Compute the relative pose from pose_a to pose_b in SE(3).
    
    Parameters:
        pose_a: (4, 4) SE(3) matrix
        pose_b: (4, 4) SE(3) matrix
        
    Returns:
        bTa: (4, 4) SE(3) matrix
    """
    # return np.linalg.inv(pose_a) @ pose_b
    wRa = pose_a[:3, :3]
    wRb = pose_b[:3, :3]
    wta = pose_a[:3, 3].reshape(-1, 1)
    wtb = pose_b[:3, 3].reshape(-1, 1)
    bRa = wRb.T @ wRa
    bta = (wRb.T @ (wta - wtb)).squeeze()
    bTa = np.eye(4)
    bTa[:3, :3] = bRa
    bTa[:3, 3] = bta
    return bTa

def find_closest_timestamps_indices(frame_ts, sensor_ts):
    # Ensure inputs are numpy arrays and sorted
    frame_ts = np.asarray(frame_ts)
    sensor_ts = np.asarray(sensor_ts)
    
    # Find insertion points for frame timestamps in sensor timestamps
    idx = np.searchsorted(sensor_ts, frame_ts)
    
    # Handle edge cases
    idx = np.clip(idx, 1, len(sensor_ts) - 1)
    
    # Compare distances to left and right neighbors
    left_distances = np.abs(sensor_ts[idx-1] - frame_ts)
    right_distances = np.abs(sensor_ts[idx] - frame_ts)
    
    # Select closest timestamp
    closest_idx = idx - (left_distances < right_distances)
    
    return closest_idx

def calculate_cos_angle(v1, v2):
    cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
    return cos_angle

def is_image_blurry_fft(image: np.ndarray, threshold: float = 10.0, viz: bool = False) -> bool:
    """
    Determine if an image is blurry using FFT.

    Parameters:
    - image (np.ndarray): Input image as a NumPy array (grayscale or color).
    - threshold (float): Threshold to classify an image as blurry.
      Lower values indicate more sensitivity to blur.
    - viz (bool): Whether to visualize the FFT spectrum.

    Returns:
    - bool: True if the image is blurry, False otherwise.
    """
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        grayscale_image = image

    # Apply FFT to the image
    f = np.fft.fft2(grayscale_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)

    # Calculate the high-frequency energy
    rows, cols = grayscale_image.shape
    crow, ccol = rows // 2, cols // 2

    # Define a small low-frequency region (center)
    low_freq_radius = min(rows, cols) // 16
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), low_freq_radius, 0, -1)

    # Mask the low frequencies to isolate high frequencies
    high_freq_energy = magnitude_spectrum * mask

    # Sum up the high-frequency energy values
    high_freq_sum = np.sum(high_freq_energy)

    # Normalize and threshold the energy to classify blur
    normalized_energy = high_freq_sum / (rows * cols)

    # Visualize the spectrum if requested
    if viz:
        fig = plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title(f"FFT Magnitude Spectrum,\n Normalized Energy: {normalized_energy:.2f}")
        plt.imshow(magnitude_spectrum, cmap='viridis')
        plt.colorbar()
        plt.axis('off')

        plt.savefig('fft_spectrum.png')
        plt.close()

    return normalized_energy < threshold

def load_and_process_poses(data_path: Path):
    """Load odometry and IMU data and create SE(3) poses."""
    # Load odometry data
    odom_data = pd.read_csv(data_path / 'odometry.csv')
    imu_data = pd.read_csv(data_path / 'imu.csv')
    odom_data.columns = odom_data.columns.str.strip()
    imu_data.columns = imu_data.columns.str.strip()

    # Create rotation matrices from quaternions
    rotations = R.from_quat(odom_data[['qx', 'qy', 'qz', 'qw']].values, scalar_first=False).as_matrix()
    q_prime = R.from_quat(np.array([[1., 0., 0., 0.]]), scalar_first=False).as_matrix()
    rotations = rotations @ q_prime

    # Extract translations
    translations = odom_data[['x', 'y', 'z']].values

    # Create SE(3) matrices
    num_poses = len(odom_data)
    poses = np.zeros((num_poses, 4, 4))
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = translations
    poses[:, 3, 3] = 1  # Homogeneous coordinates

    return poses, odom_data, imu_data

def arkit_to_opencv(poses: np.ndarray):
    """Apply pose convention transformation from ARKit to OpenCV."""
    # Transformation matrix for flipping the y and z-axis
    T_flip = np.array([
            [1,  0,  0,  0],
            [0, -1,  0,  0],
            [0,  0, -1,  0],
            [0,  0,  0,  1]
        ])
    # Apply the transformation to all poses
    poses = T_flip[None] @ poses @ np.linalg.inv(T_flip)[None]
    return poses

def plot_poses_and_imu(poses: np.ndarray, imu_data: pd.DataFrame, plot_poses: bool, plot_imu: bool):
    """Handle plotting of poses and IMU data."""
    if plot_poses:
        plot_SE3(poses, title="Pose Visualization", color='blue', symbol='frames', show_sec=True, params={'scale': 0.01, 'width': 5})

    if plot_imu:
        # Adjust timestamps to start from 0
        imu_ts_shifted = imu_data['timestamp'] - imu_data['timestamp'].iloc[0]

        # Create subplots with shared x-axis
        fig = sp.make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.1, 
                               subplot_titles=['a_x', 'alpha_x', 'a_y', 'alpha_y', 'a_z', 'alpha_z'])

        # Add traces for each column
        for i, column in enumerate(['a_x', 'a_y', 'a_z'], start=1):
            fig.add_trace(
                go.Scatter(
                    x=imu_ts_shifted,
                    y=imu_data[column],
                    mode='lines',
                    name=column
                ),
                row=i, col=1
            )
        for i, column in enumerate(['alpha_x', 'alpha_y', 'alpha_z'], start=1):
            fig.add_trace(
                go.Scatter(
                    x=imu_ts_shifted,
                    y=imu_data[column],
                    mode='lines',
                    name=column
                ),
                row=i, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="IMU Data Over Time"
        )

        # Show the plot
        fig.show()

def extract_frames_from_video(data_path: Path):
    """Extract frames from video using ffmpeg."""
    os.makedirs(os.path.join(data_path, 'rgb'), exist_ok=True)
    cmd = f'ffmpeg -i {data_path}/rgb.mp4 -start_number 0 -q:v 2 {data_path}/rgb/%06d.jpg'
    os.system(cmd)

    # Assert that the frame shapes are 1920x1440
    frame_shapes = set()
    for frame_path in (data_path / 'rgb').iterdir():
        frame = Image.open(frame_path)
        frame_shapes.add(frame.size)
    assert len(frame_shapes) == 1, f"Frame shapes are not consistent: {frame_shapes}"
    assert frame_shapes.pop() == (1920, 1440), f"Frame shapes are not 1920x1440: {frame_shapes}"

def process_images(data_path: Path, resize_dims: list, crop_dims: list, rgb_suffix: str):
    """Handle image resizing, cropping, and camera matrix updates."""
    if not (any(np.array(resize_dims) > 0) or any(np.array(crop_dims) > 0)):
        return

    suffix = rgb_suffix if (rgb_suffix[0] == '_' or rgb_suffix == '') else '_' + rgb_suffix
    output_dir = data_path / f'rgb{suffix}'
    os.makedirs(output_dir, exist_ok=True)
    
    if output_dir == data_path / 'rgb':
        cont = input(f"{colors.WARNING}WARNING: This will overwrite the original rgb directory. Is this intended? (y/n){colors.ENDC}")
        if cont not in ['y', 'Y', 'yes', 'Yes', 'YES']:
            print("Aborting..")
            exit()

    # Copy original images to new directory if using suffix
    if rgb_suffix:
        os.system(f'cp {data_path}/rgb/*.jpg {output_dir}/')

    # Load the camera matrix
    camera_matrix_path = data_path / 'camera_matrix.csv'
    camera_matrix = np.loadtxt(camera_matrix_path, delimiter=',')

    # Original dimensions
    original_dims = np.array([1920, 1440])  # Width, Height
    current_dims = original_dims.copy()

    if any(np.array(resize_dims) > 0):
        print(f"{colors.OKCYAN}Resizing frames to {resize_dims}{colors.ENDC}")
        w, h = resize_dims
        resize_transform = Resize((h, w), antialias=True)

        # Process each image
        for img_path in (data_path / 'rgb').glob('*.jpg'):
            img = read_image(str(img_path))
            resized_img = resize_transform(img)
            write_jpeg(resized_img, str(output_dir / img_path.name), quality=100)

        # Update camera matrix for resizing
        scale_factors = np.array(resize_dims) / original_dims
        camera_matrix[0, 0] *= scale_factors[0]  # fx'
        camera_matrix[1, 1] *= scale_factors[1]  # fy'
        camera_matrix[0, 2] *= scale_factors[0]  # cx'
        camera_matrix[1, 2] *= scale_factors[1]  # cy'
        current_dims = np.array(resize_dims)

    if any(np.array(crop_dims) > 0):
        print(f"{colors.OKCYAN}Cropping frames to {crop_dims}{colors.ENDC}")
        w, h = crop_dims

        # Calculate crop parameters
        crop_x = (current_dims[0] - w) // 2
        crop_y = (current_dims[1] - h) // 2

        center_crop_transform = CenterCrop((h, w))

        # Process each image
        for img_path in output_dir.glob('*.jpg'):
            img = read_image(str(img_path))
            cropped_img = center_crop_transform(img)
            write_jpeg(cropped_img, str(img_path), quality=100)

        # Adjust principal points by subtracting the crop offset
        camera_matrix[0, 2] -= crop_x  # cx'
        camera_matrix[1, 2] -= crop_y  # cy'

    # Save the updated camera matrix
    updated_camera_matrix_path = data_path / f'camera_matrix{suffix}.csv'
    calib_param_path = data_path / f'calib_params{suffix}.txt'
    np.savetxt(updated_camera_matrix_path, camera_matrix, delimiter=',', fmt='%.6f')
    with open(calib_param_path, 'w') as f:
        f.write(f"{camera_matrix[0, 0]} {camera_matrix[1, 1]} {camera_matrix[0, 2]} {camera_matrix[1, 2]}")
    print(f"{colors.OKGREEN}Updated camera matrix saved to {updated_camera_matrix_path}{colors.ENDC}")

def compute_trajectory_velocities(poses: np.ndarray, odom_data: pd.DataFrame):
    """Calculate relative poses and velocities from trajectory data."""
    trajectory_aTbs = []
    trans_vels = []
    rot_vels = []
    timestamps = odom_data['timestamp'].values
    exposures = odom_data['exposure'].values
    timestamps = (timestamps - timestamps[1])[1:] # The second timestamp is what's aligned with the first RGB frame
    frame_gaps = timestamps[1:] - timestamps[:-1]
    
    for i in range(1, len(poses) - 1):
        pose_prev = poses[i - 1]
        pose_next = poses[i + 1]

        alpha = exposures[i] / (2*frame_gaps[i-1])

        # Calculate the relative pose
        bTa = calculate_relative_pose(pose_prev, pose_next)
        aTb = np.linalg.inv(bTa)

        # Scale the relative pose by the exposure time
        scaled_aTb = interpolate_pose(np.eye(4), aTb, alpha)

        trajectory_aTbs.append(scaled_aTb)
        trans_vels.append(scaled_aTb[:3, 3] / exposures[i])
        M = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
        rot_vels.append((R.from_matrix(M @ scaled_aTb[:3, :3] @ M.T).as_euler('xyz', degrees=False)) / exposures[i])

    # Repeat the last rel pose and vel to match the number of frames
    trajectory_aTbs.append(trajectory_aTbs[-1])
    trans_vels.append(trans_vels[-1])
    rot_vels.append(rot_vels[-1])
    trans_vels = np.array(trans_vels)
    rot_vels = np.array(rot_vels)

    return trajectory_aTbs, trans_vels, rot_vels, timestamps

def detect_and_save_blurry_frames(data_path: Path, trajectory_aTbs: list, timestamps: np.ndarray, 
                                  odom_data: pd.DataFrame, blurry_config: dict):
    """Detect and save blurry frames based on configuration."""
    if not blurry_config['ENABLED']:
        return

    blurred = []
    ts1, ts2 = [], []
    bRa_qx, bRa_qy, bRa_qz, bRa_qw = [], [], [], []
    bta_x, bta_y, bta_z = [], [], []
    blurred_exposures = []
    exposures = odom_data['exposure'].values
    
    assert blurry_config['USE_SUFFIX'].split('_')[-1] == '320x224', "Currently only supports 320x224 frames"
    blurry_dir = data_path / f'blurry_frames{blurry_config["USE_SUFFIX"]}'
    blurry_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate over all the frames
    for i, img_path in enumerate(sorted((data_path / ('rgb' + blurry_config['USE_SUFFIX'])).iterdir())):
        image = cv2.imread(str(img_path))
        if blurry_config['METHOD'] == "fft":
            is_blurry = is_image_blurry_fft(image, threshold=blurry_config['THRESHOLD'], viz=blurry_config['VIZ'])
        elif blurry_config['METHOD'] == "laplacian":
            raise NotImplementedError("Laplacian method not implemented")
        else:
            raise ValueError(f"Unknown blurry detection method: {blurry_config['METHOD']}")
            
        if is_blurry:
            # Save the image to the blurry directory
            shutil.copy(img_path, blurry_dir / img_path.name)
            bRa = trajectory_aTbs[i][:3, :3].transpose()
            bta = (bRa @ -trajectory_aTbs[i][:3, 3][:, None]).squeeze()
            bRa_qvec = R.from_matrix(bRa).as_quat(scalar_first=False)
            blurred.append(blurry_dir / img_path.name)
            ts1.append(timestamps[i])
            ts2.append(timestamps[i])
            bRa_qx.append(bRa_qvec[0])
            bRa_qy.append(bRa_qvec[1])
            bRa_qz.append(bRa_qvec[2])
            bRa_qw.append(bRa_qvec[3])
            bta_x.append(bta[0])
            bta_y.append(bta[1])
            bta_z.append(bta[2])
            blurred_exposures.append(exposures[i])
            
    # Create a dataframe from the blurred frames
    camera_matrix_path = data_path / f'camera_matrix{blurry_config["USE_SUFFIX"]}.csv'
    camera_matrix = np.loadtxt(camera_matrix_path, delimiter=',')
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    blurred_df = pd.DataFrame({
        'blurred': blurred,
        'ts1': ts1,
        'ts2': ts2,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'bRa_qx': bRa_qx,
        'bRa_qy': bRa_qy,
        'bRa_qz': bRa_qz,
        'bRa_qw': bRa_qw,
        'bta_x': bta_x,
        'bta_y': bta_y,
        'bta_z': bta_z,
        'exposure': blurred_exposures
    })
    blurred_df.to_csv(data_path / f'blurred_frames{blurry_config["USE_SUFFIX"]}.csv', index=False)

def plot_velocities(trans_vels: np.ndarray, timestamps: np.ndarray, imu_data: pd.DataFrame, 
                   closest_idx: np.ndarray, imu_aligned_ts: np.ndarray):
    """Plot translational and angular velocities."""
    # Define the columns to plot
    t_columns = ['t_x', 't_y', 't_z']
    alpha_columns = ['alpha_x', 'alpha_y', 'alpha_z']

    # Create subplots with shared x-axis
    fig = sp.make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=['t_x', 'alpha_x', 't_y', 'alpha_y', 't_z', 'alpha_z'])

    # Add traces for each column
    for i, column in enumerate(t_columns, start=1):
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=trans_vels[:, i-1],
                mode='lines',
                name=column
            ),
            row=i, col=1
        )

    for i, column in enumerate(alpha_columns, start=1):
        fig.add_trace(
            go.Scatter(
                x=imu_aligned_ts,
                y=imu_data[column][closest_idx],
                mode='lines',
                name=column
            ),
            row=i, col=2
        )

    # Update layout
    fig.update_layout(
        title_text="Translational and Angular Velocities"
    )

    # Show the plot
    fig.show()

def save_arkit_poses(data_path: Path, poses: np.ndarray, odom_data: pd.DataFrame):
    """Save ARKit poses."""
    rotations_quat = R.from_matrix(poses[:,:3,:3]).as_quat(scalar_first=False)
    arkit_poses_df = pd.DataFrame({
        'timestamp': odom_data['timestamp'].values[1:],
        't_x': poses[1:, 0, 3],
        't_y': poses[1:, 1, 3],
        't_z': poses[1:, 2, 3],
        'qx': rotations_quat[1:, 0],
        'qy': rotations_quat[1:, 1],
        'qz': rotations_quat[1:, 2],
        'qw': rotations_quat[1:, 3]
    })
    arkit_poses_df.to_csv(data_path / 'arkit_world_poses.csv', index=False)

def save_final_outputs(data_path: Path, arkit_poses: np.ndarray, opencv_poses: np.ndarray, odom_data: pd.DataFrame, 
                      trans_vels: np.ndarray, imu_data: pd.DataFrame, closest_idx: np.ndarray):
    """Save final velocity and OpenCV pose data to CSV files."""
    # Save velocities
    aligned_alpha_x = imu_data['alpha_x'].values[closest_idx]
    aligned_alpha_y = imu_data['alpha_y'].values[closest_idx]
    aligned_alpha_z = imu_data['alpha_z'].values[closest_idx]
    
    velocities_df = pd.DataFrame({
        'timestamp': odom_data['timestamp'].values[1:],
        't_x': trans_vels[:, 0],
        't_y': trans_vels[:, 1],
        't_z': trans_vels[:, 2],
        'alpha_x': aligned_alpha_x,
        'alpha_y': aligned_alpha_y,
        'alpha_z': aligned_alpha_z
    })
    velocities_df.to_csv(data_path / 'velocities.csv', index=False)

    arkit_quat = R.from_matrix(arkit_poses[:,:3,:3]).as_quat(scalar_first=False)
    arkit_poses_df = pd.DataFrame({
        'timestamp': odom_data['timestamp'].values[1:],
        't_x': arkit_poses[1:, 0, 3],
        't_y': arkit_poses[1:, 1, 3],
        't_z': arkit_poses[1:, 2, 3],
        'qx': arkit_quat[1:, 0],
        'qy': arkit_quat[1:, 1],
        'qz': arkit_quat[1:, 2],
        'qw': arkit_quat[1:, 3]
    })
    arkit_poses_df.to_csv(data_path / 'arkit_world_poses.csv', index=False)

    # Save OpenCV poses
    opencv_quat = R.from_matrix(opencv_poses[:,:3,:3]).as_quat(scalar_first=False)
    opencv_poses_df = pd.DataFrame({
        'timestamp': odom_data['timestamp'].values[1:],
        't_x': opencv_poses[1:, 0, 3],
        't_y': opencv_poses[1:, 1, 3],
        't_z': opencv_poses[1:, 2, 3],
        'qx': opencv_quat[1:, 0],
        'qy': opencv_quat[1:, 1],
        'qz': opencv_quat[1:, 2],
        'qw': opencv_quat[1:, 3]
    })
    opencv_poses_df.to_csv(data_path / 'opencv_poses.csv', index=False)