import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
from torchvision.utils import make_grid, save_image
from matplotlib.colors import hsv_to_rgb
from iaai.utils.metrics import find_closest_timestamps_indices

def postprocess_batch_img_visualization(img_BCHW):
    # Unnormalize using ImageNet statistics
    device = img_BCHW.device
    img_BCHW = img_BCHW * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_BCHW = img_BCHW + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    img_BCHW = torch.clamp(img_BCHW, 0, 1)
    img_BCHW = (img_BCHW * 255).to(torch.uint8)
    return img_BCHW


def create_hsv_plot_from_offsets(offsets_2hw, max_pxl_displ=20., upscale_term=14, mod_half=False):
    """
    Generates an HSV image from the 2-channel (h, w) PyTorch tensor of offsets.
    
    Parameters:
        offsets_2hw (torch.Tensor): A (2, h, w) tensor where each pixel has 
                                    (dx, dy) offset coordinates.
                                   
    Returns:
        RGB (torch.Tensor): An (14h, 14w, 3) RGB image representing the upscaled vector field.
    """
    # Compute orientation (Hue) in the range [0, 1]
    if mod_half:
        orientation = torch.atan2(offsets_2hw[1], offsets_2hw[0]) % torch.pi # Invariant to direction flips
        H = orientation / torch.pi
    else:
        orientation = torch.atan2(offsets_2hw[1], offsets_2hw[0])
        H = (orientation + torch.pi) / (2 * torch.pi)

    # Compute magnitude (Saturation)
    magnitude = torch.sqrt(offsets_2hw[0]**2 + offsets_2hw[1]**2)  # Magnitude of the vector
    S = magnitude / max_pxl_displ
    S = torch.clamp(S, 0, 1)  # Clamp values

    # Set Value (V) to constant 1
    V = torch.ones_like(H)

    # Stack H, S, V to create an HSV image (h, w, 3)
    HSV = torch.stack((H, S, V), dim=-1)  # Shape will be (h, w, 3)

    # Upscale the HSV image by a factor of 14
    h, w = H.shape
    HSV = HSV.repeat_interleave(upscale_term, dim=0).repeat_interleave(upscale_term, dim=1)
    HSV = HSV.detach().cpu().numpy()

    # Convert HSV to RGB for visualization
    RGB = torch.from_numpy(hsv_to_rgb(HSV)).permute(2, 0, 1)

    return RGB


def visualize_quivers(blurred_img_BCHW, gt_offsets_B2hw, pred_offsets_B2hw, num_images=4, subsample_factor=14):
    def create_quivers_plot(offsets_N2, img_np, color='r'):
        # Create a quiver plot on the provided image
        fig, ax = plt.subplots(figsize=(5, 5))
        plt.clf()
        plt.imshow(img_np)
        plt.quiver(
            X.flatten().detach().cpu().numpy(),
            Y.flatten().detach().cpu().numpy(),
            offsets_N2[:, 0].detach().cpu().numpy(),
            offsets_N2[:, 1].detach().cpu().numpy(),
            color=color,
            scale=1.,
            scale_units='xy',
            angles='xy'
        )
        plt.axis('off')
        plt.tight_layout()

        # Save plot to a NumPy array
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img_data = img_data[..., :3]  # Remove alpha channel
        plt.close(fig)
        return img_data

    h, w = gt_offsets_B2hw.shape[-2:]
    gt_offsets_B2hw = gt_offsets_B2hw[..., subsample_factor//2:h + subsample_factor//2:subsample_factor, subsample_factor//2:w + subsample_factor//2:subsample_factor]
    pred_offsets_B2hw = pred_offsets_B2hw[..., subsample_factor//2:h + subsample_factor//2:subsample_factor, subsample_factor//2:w + subsample_factor//2:subsample_factor]

    # Get num_images equally spaced images
    inds = torch.linspace(0, gt_offsets_B2hw.shape[0] - 1, num_images, dtype=int)
    pred_offsets_b2hw = pred_offsets_B2hw[inds]
    gt_offsets_b2hw = gt_offsets_B2hw[inds]
    blurred_imgs_bchw = postprocess_batch_img_visualization(blurred_img_BCHW[inds])

    # Create mesh grid for quiver based on actual subsampled offset dimensions
    num_patches_h, num_patches_w = gt_offsets_B2hw.shape[-2:]
    i_centers = torch.arange(num_patches_h) * subsample_factor + subsample_factor//2
    j_centers = torch.arange(num_patches_w) * subsample_factor + subsample_factor//2
    X, Y = torch.meshgrid(j_centers, i_centers, indexing='xy')

    # Collect all quiver plots into a grid
    quiver_images = []
    for img, pred_offset, gt_offset in zip(blurred_imgs_bchw, pred_offsets_b2hw, gt_offsets_b2hw):
        # Prepare image for plotting
        img_np = img.clone().permute(1, 2, 0).cpu().numpy()

        # Predicted quiver (red)
        pred_quiver_img = create_quivers_plot(pred_offset.permute(1,2,0).reshape(-1, 2), img_np, color='r')

        # Ground truth quiver (green)
        gt_quiver_img = create_quivers_plot(gt_offset.permute(1,2,0).reshape(-1, 2), img_np, color='g')

        quiver_images.append((pred_quiver_img, gt_quiver_img))

    # Combine images into a grid
    quiver_grid = np.vstack([
        np.hstack([quiver[0] for quiver in quiver_images]),  # Top row: predicted
        np.hstack([quiver[1] for quiver in quiver_images])   # Bottom row: ground truth
    ])

    return quiver_grid


def visualize_hsv(gt_offsets_B2hw, pred_offsets_B2hw, upscale_term=14, num_images=4, mod_half=False):
    # Obtain subset of the images, b = num_images
    # Get num_images equally spaced images
    inds = torch.linspace(0, gt_offsets_B2hw.shape[0]-1, num_images, dtype=int)
    pred_offsets_b2hw = pred_offsets_B2hw[inds]
    gt_offsets_b2hw = gt_offsets_B2hw[inds]

    # Create HSV plots for each image
    hsv_images = []
    for i in range(num_images):
        hsv_image = create_hsv_plot_from_offsets(pred_offsets_b2hw[i], upscale_term=upscale_term, mod_half=mod_half)
        hsv_images.append(hsv_image)
    for i in range(num_images):
        hsv_image = create_hsv_plot_from_offsets(gt_offsets_b2hw[i], upscale_term=upscale_term, mod_half=mod_half)
        hsv_images.append(hsv_image)

    grid = make_grid(hsv_images, nrow=num_images)

    return grid

def visualize_depth_maps(gt_depth_BHW, pred_depth_BHW, blurred_img_BCHW, min_depth=0.0, max_depth=10.0, num_images=4):
    # Obtain subset of the images, b = num_images
    # Get num_images equally spaced images
    inds = torch.linspace(0, gt_depth_BHW.shape[0]-1, num_images, dtype=int)
    gt_depth_bHW = gt_depth_BHW[inds]
    pred_depth_bHW = pred_depth_BHW[inds]
    blurred_img_bCHW = postprocess_batch_img_visualization(blurred_img_BCHW[inds])

    # Normalize depth map to be between 0 and 1
    gt_depth_norm_bHW = (gt_depth_bHW - min_depth) / (max_depth - min_depth)
    gt_depth_norm_bHW = torch.clamp(gt_depth_norm_bHW, 0, 1)
    gt_depth_norm_bHW = gt_depth_norm_bHW.cpu().numpy()
    pred_depth_norm_bHW = (pred_depth_bHW - min_depth) / (max_depth - min_depth)
    pred_depth_norm_bHW = torch.clamp(pred_depth_norm_bHW, 0, 1)
    pred_depth_norm_bHW = pred_depth_norm_bHW.cpu().numpy()

    # Create single figure with 3 rows
    fig = plt.figure(figsize=(4*num_images, 12))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    for i in range(num_images):
        # Plot blurred image
        plt.subplot(3, num_images, i+1)
        plt.imshow(blurred_img_bCHW[i].permute(1, 2, 0).cpu().numpy())
        plt.axis('off')

        # Plot predicted depth
        plt.subplot(3, num_images, num_images+i+1)
        plt.imshow(pred_depth_norm_bHW[i], cmap='Spectral')
        plt.axis('off')
        
        # Plot ground truth depth
        plt.subplot(3, num_images, 2*num_images+i+1)
        plt.imshow(gt_depth_norm_bHW[i], cmap='Spectral')
        plt.axis('off')

    return fig

def viz_time_series(pred_vels_N3, gt_vels_N3, pred_ts, gt_ts, rot_or_trans="rot", title="", save_path=None):
    """Visualize time series comparison between predicted and ground truth velocities."""
    closest_indices = find_closest_timestamps_indices(pred_ts, gt_ts)
    closest_gt_ts = gt_ts[closest_indices]
    abs_error = np.abs(pred_vels_N3 - gt_vels_N3[closest_indices])

    fig = sp.make_subplots(rows=3, cols=2)
    if rot_or_trans == "rot":
        dim_names = [r'$\omega_x \text{ (s}^{-1}\text{)}$', r'$\omega_y \text{ (s}^{-1}\text{)}$', r'$\omega_z \text{ (s}^{-1}\text{)}$']
    elif rot_or_trans == "trans":
        dim_names = [r'$v_x \text{ (m/s)}$', r'$v_y \text{ (m/s)}$', r'$v_z \text{ (m/s)}$']
    else:
        raise ValueError(f"Invalid rot_or_trans: {rot_or_trans}")

    for i, dim_name in enumerate(dim_names):
        fig.add_trace(
            go.Scatter(
                x=gt_ts - min(pred_ts[0], gt_ts[0]),
                y=gt_vels_N3[:, i], 
                name="Ground Truth",
                fill=None, 
                line=dict(color="blue"),
                showlegend=(i == 0)
            ),
            row=i+1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=pred_ts - min(pred_ts[0], gt_ts[0]), 
                y=pred_vels_N3[:, i], 
                name="Predicted",
                fill='tonexty', 
                marker=dict(color="red"),
                showlegend=(i == 0)
            ),
            row=i+1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=closest_gt_ts - min(pred_ts[0], gt_ts[0]), 
                y=abs_error[:, i], 
                name="Abs Error", 
                mode="markers",
                marker=dict(color="red"),
                showlegend=(i == 0)
            ),
            row=i+1, col=2
        )
        fig.update_xaxes(title_text="Time (s)", row=i+1, col=1)
        fig.update_yaxes(title_text=dim_name, row=i+1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=i+1, col=2)
        fig.update_yaxes(title_text=f"Abs Error", row=i+1, col=2)

    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    if save_path is not None:
        fig.write_image(save_path)
    fig.show()
