import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_flow(rgb, flow_hw2, backbone_name, save_path="flow_viz.png"):
    """Visualize optical flow on RGB image."""
    if backbone_name == "segnext":
        flow_hw2 = flow_hw2[7:252-7:14, 7:336-7:14]
    motion_displ_n2 = flow_hw2.reshape(-1, 2)
    h, w, _ = flow_hw2.shape
    X, Y = torch.meshgrid(torch.arange(7, w*14, 14), torch.arange(7, h*14, 14))

    plt.clf()
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    rgb_np = (rgb*255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    axes.imshow(rgb_np)
    axes.quiver(
        X.flatten().detach().cpu().numpy(),
        Y.flatten().detach().cpu().numpy(),
        motion_displ_n2[:, 0].detach().cpu().numpy(),
        motion_displ_n2[:, 1].detach().cpu().numpy(),
        color='r',
        scale=1.0,
        scale_units='xy',
        angles='xy'
    )
    axes.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    
    fig.canvas.draw()
    img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_data = img_data[..., :3]
    plt.close(fig)

    return img_data

def visualize_depth(depth_HW, save_path='depth_viz.png'):
    """Visualize depth map using a heatmap colorscheme."""
    depth_norm = (depth_HW - depth_HW.min()) / (depth_HW.max() - depth_HW.min())
    depth_colored = plt.cm.Spectral(depth_norm)
    
    plt.figure()
    plt.imshow(depth_colored)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
