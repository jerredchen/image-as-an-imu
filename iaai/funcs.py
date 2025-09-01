import torch
import roma

def compute_photometric_error(
        flow_HW2, 
        source_img, 
        target_img,
        framerate_dt=1/30,
        exposure_time_s=0.01,
        device="cuda"
    ):
    """Compute photometric error between warped source image and target image.
    
    Args:
        flow_HW2: (2,H,W) tensor of flow field computed during exposure time
        source_img: (3,H,W) tensor of source image to warp
        target_img: (3,H,W) tensor of target image to compare against
    Returns:
        error: float, mean photometric error between warped and target images
        warped_img: warped source image
    """
    # Scale flow from exposure time to frame time
    # Negate flow because grid_sample performs backwards warping
    scaled_flow_HW2 = -(framerate_dt / exposure_time_s) * flow_HW2
    
    # Create sampling grid for warping
    H, W = flow_HW2.shape[:-1]
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    
    # Add flow to base coordinates and normalize to [-1, 1] for grid_sample
    flow_y, flow_x = scaled_flow_HW2[..., 1], scaled_flow_HW2[..., 0]
    sample_y = 2.0 * (y_coords + flow_y) / (H - 1) - 1.0
    sample_x = 2.0 * (x_coords + flow_x) / (W - 1) - 1.0
    sampling_grid = torch.stack([sample_x, sample_y], dim=-1)
    sampling_grid = sampling_grid[::10,::10] # Downsample by 10x
    
    # Warp image using grid_sample
    warped_img = torch.nn.functional.grid_sample(
        source_img.unsqueeze(0),  # Add batch dimension
        sampling_grid.unsqueeze(0),  # Add batch dimension
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(0)  # Remove batch dimension

    # Compute photometric error (L1 distance)
    valid_mask = (sample_x >= -1) & (sample_x <= 1) & (sample_y >= -1) & (sample_y <= 1)
    error = torch.abs(warped_img - target_img[:, ::10,::10])
    error = error.mean(dim=0)  # Average across channels
    error = error[valid_mask[::10, ::10]].mean()  # Average across valid pixels
    
    return error.item(), warped_img

def disambiguate_direction(
        flow_HW2,
        exposure_time_s, 
        framerate_dt, 
        aRb, 
        atb,
        rgb1_CHW,
        rgb2_CHW,
        rgb0_CHW=None,
        M=None
    ):
    """Disambiguate motion direction by comparing photometric errors.
    
    Args:
        flow_HW2: (2,H,W) tensor of flow field
        exposure_time_s: float, exposure time in seconds
        framerate_dt: float, time between frames in seconds
        aRb: (3,3) tensor of rotation matrix
        atb: (3,1) tensor of translation vector
        rgb1_CHW: (3,H,W) tensor of first RGB image
        rgb2_CHW: (3,H,W) tensor of second RGB image
        rgb0_CHW: (3,H,W) tensor of previous RGB image (optional)
        M: (3,3) transformation matrix to apply to aRb. If None, is identity.
    Returns:
        rot_vel: (3,) tensor of disambiguated rotational velocity in rad/s
        trans_vel: (3,) tensor of disambiguated translational velocity in m/s
    """
    device = aRb.device
    H, W = flow_HW2.shape[:-1]

    fw_flow_HW2 = flow_HW2.clone()
    bw_flow_HW2 = -(flow_HW2.clone())
    
    # Check forward and backward flow for current->next frame
    fw_error_next, fw_warped_img = compute_photometric_error(
        fw_flow_HW2, rgb1_CHW, rgb2_CHW, framerate_dt, exposure_time_s, device)
    bw_error_next, bw_warped_img = compute_photometric_error(
        bw_flow_HW2, rgb1_CHW, rgb2_CHW, framerate_dt, exposure_time_s, device)
    
    # Initialize variables for previous frame check
    fw_error_prev, bw_error_prev = float('inf'), float('inf')
    
    # If previous frame is available, check forward and backward flow for prev->current frame
    if rgb0_CHW is not None:
        fw_error_prev, _ = compute_photometric_error(
            fw_flow_HW2, rgb1_CHW, rgb0_CHW, framerate_dt, exposure_time_s, device)
        bw_error_prev, _ = compute_photometric_error(
            bw_flow_HW2, rgb1_CHW, rgb0_CHW, framerate_dt, exposure_time_s, device)
    
    # Determine which direction has lower error
    # For next frame: lower fw_error_next means forward motion is better
    # For prev frame: lower bw_error_prev means forward motion is better (since we're looking backward)
    fw_total_error = fw_error_next + (bw_error_prev if rgb0_CHW is not None else 0)
    bw_total_error = bw_error_next + (fw_error_prev if rgb0_CHW is not None else 0)
    
    # Apply transformation based on which direction has lower error
    # M = torch.tensor([[ 0, -1,  0],
    #                   [-1,  0,  0],
    #                   [ 0,  0, -1]]).float()
    if M is None:
        M = torch.eye(3)
    M = M.float().to(device)

    if fw_total_error < bw_total_error:
        aRb = M @ aRb @ M.T
    else:
        atb = -aRb.T @ atb
        aRb = M @ aRb.T @ M.T
    
    return roma.rotmat_to_euler('XYZ', aRb) / exposure_time_s, atb.squeeze() / exposure_time_s

def compute_single_image_velocity(pose_6D, exposure_time_s, M=None, device="cpu"):
    """Compute velocity from pose.
    
    Args:
        pose_6D: (6,) tensor of pose
        exposure_time_s: float, exposure time in seconds
        M: (3,3) transformation matrix to apply to aRb. If None, is identity.
    Returns:
        vel_B3: (B,3) tensor of velocity
    """
    aRb = roma.euler_to_rotmat('XYZ', pose_6D[:3]).squeeze().to(device)
    atb = pose_6D[3:].squeeze().to(device)

    if M is None:
        M = torch.eye(3)
    M = M.float().to(device)
    aRb = M @ aRb @ M.T

    rot_vel = roma.rotmat_to_euler('XYZ', aRb) / exposure_time_s
    trans_vel = atb.squeeze() / exposure_time_s

    return rot_vel, trans_vel