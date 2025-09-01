from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor, normalize, center_crop, resize

from .modules import MSCAN, ConvModule, LightHamHead

class LowLevelEncoder(nn.Module):
    """Very simple low-level encoder."""

    def __init__(self):
        """
        Simple low-level encoder.
        Adapted from GeoCalib: https://github.com/cvg/GeoCalib/blob/main/geocalib/geocalib.py
        """
        super().__init__()
        self.in_channel = 3
        self.feat_dim = 64

        self.conv1 = ConvModule(self.in_channel, self.feat_dim, kernel_size=3, padding=1)
        self.conv2 = ConvModule(self.feat_dim, self.feat_dim, kernel_size=3, padding=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = data["image"]

        assert (
            x.shape[-1] % 32 == 0 and x.shape[-2] % 32 == 0
        ), "Image size must be multiple of 32 if not using single image input."

        c1 = self.conv1(x)
        c2 = self.conv2(c1)

        return {"features": c2}

class FlowDecoder(nn.Module):
    """
    Small flow decoder. Follows decoder from GeoCalib:
    https://github.com/cvg/GeoCalib/blob/main/geocalib/geocalib.py
    """

    def __init__(self):
        """Flow decoder."""
        super().__init__()
        self.decoder = LightHamHead()
        self.linear_pred_flow = nn.Conv2d(self.decoder.out_channels, 2, kernel_size=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.decoder(data["features"])
        flow = self.linear_pred_flow(x)

        return {"flow_field": flow}

class DepthDecoder(nn.Module):
    """
    Small depth decoder. Follows decoder from GeoCalib:
    https://github.com/cvg/GeoCalib/blob/main/geocalib/geocalib.py
    """

    def __init__(self):
        """Depth decoder."""
        super().__init__()
        self.decoder = LightHamHead()
        self.linear_pred_depth = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        x = self.decoder(data["features"])
        depth = self.linear_pred_depth(x).exp().clip(min=1e-8, max=torch.inf)

        return {"depth": depth}

class PoseHead(nn.Module):
    def __init__(
        self,
        use_pinv=False,
        cut_border_amnt=20,
        sample_n=10,
    ):
        super().__init__()
        self.use_pinv = use_pinv
        self.cut_border_amnt = cut_border_amnt
        self.sample_n = sample_n

    def compute_camera_motion(self, flow_B2HW, depths_BHW, fl_B):
        B, _, H, W = flow_B2HW.shape
        # Remove 20 pixels off the edges to avoid border effects
        if self.cut_border_amnt > 0:
            flow_BN2 = flow_B2HW[..., self.cut_border_amnt:-self.cut_border_amnt, self.cut_border_amnt:-self.cut_border_amnt].permute(0, 2, 3, 1).reshape(B, -1, 2)
            i_centers = torch.arange(self.cut_border_amnt, H-self.cut_border_amnt)
            j_centers = torch.arange(self.cut_border_amnt, W-self.cut_border_amnt)
            depths_BHW = depths_BHW[..., self.cut_border_amnt:-self.cut_border_amnt, self.cut_border_amnt:-self.cut_border_amnt]
        else:
            flow_BN2 = flow_B2HW.permute(0, 2, 3, 1).reshape(B, -1, 2)
            i_centers = torch.arange(H)
            j_centers = torch.arange(W)

        X, Y = torch.meshgrid(j_centers, i_centers, indexing='xy')
        depths_BN = depths_BHW.reshape(B, -1).cuda()

        u = (X.flatten() - W/2).unsqueeze(0).expand(B, -1).cuda()
        v = (Y.flatten() - H/2).unsqueeze(0).expand(B, -1).cuda()
        fl_BN = fl_B.unsqueeze(1).expand(-1, u.shape[1]).cuda()
        assert u.shape == v.shape == fl_BN.shape == depths_BN.shape

        # Construct A matrix [B, N, 12] using broadcasting
        A_BN12 = torch.stack([
            # omega_x            omega_y               omega_z   Tx                Ty                    Tz
            u*v/fl_BN,          -(fl_BN + u**2/fl_BN), v,        -fl_BN/depths_BN, torch.zeros_like(u), u/depths_BN,
            (fl_BN + v**2/fl_BN), -u*v/fl_BN,         -u,        torch.zeros_like(u), -fl_BN/depths_BN,  v/depths_BN
        ], dim=-1)  # [B, N, 12]

        # Select every 10th point to save on computation
        A_weighted = A_BN12[:,::self.sample_n].reshape(B, -1, 6)
        b_weighted = flow_BN2[:,::self.sample_n].reshape(B, -1).unsqueeze(-1)

        A = A_weighted.float().cuda()
        b = b_weighted.float().cuda()

        # Solve with LLS
        if self.use_pinv:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                x_B6 = torch.linalg.pinv(A) @ b
                # Compute residual
                res = torch.norm(A @ x_B6 - b, dim=1).squeeze()
        else:
            x_B6, res, rank, S = torch.linalg.lstsq(A, b, driver='gels')

        return x_B6.to(flow_BN2.dtype), res.to(flow_BN2.dtype)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        flow_B2HW = data["flow_field"]
        depth_BHW = data["depth"]
        fl_B = data["fl"]

        x_B6, res = self.compute_camera_motion(flow_B2HW, depth_BHW, fl_B)

        return {"pose": x_B6, "residual": res}

class Blur2PoseSegNeXtBackbone(nn.Module):
    def __init__(
            self,
            supervise_pose=True,
            use_pinv=False,
            cut_border_amnt=20,
            sample_n=10,
            device="cuda"
    ):
        """
        Network for estimating camera motion from a single motion-blurred image.

        Args:
            supervise_pose: Whether to supervise the pose head.
            use_pinv: Whether to use pseudo-inverse to solve the least-squares problem.
            cut_border_amnt: Amount of pixels to cut off the edges of the image.
            sample_n: Number of points to sample from the image for solving the least-squares problem.
            device: Device to run the model on.
        """
        super().__init__()

        self.supervise_pose = supervise_pose
        self.backbone = MSCAN()
        self.ll_enc = LowLevelEncoder()
        self.flow_decoder = FlowDecoder()
        self.depth_decoder = DepthDecoder()
        self.pose_head = PoseHead(use_pinv, cut_border_amnt, sample_n)
        self.device = device

    def forward(self, data):
        features = {"hl": self.backbone(data)["features"], "ll": self.ll_enc(data)["features"]}
        flow_out = self.flow_decoder({"features": features})
        depth_out = self.depth_decoder({"features": features})

        out = {
            "flow_field": flow_out["flow_field"],
            "depth": depth_out["depth"],
            "pose": None,
            "residual": None
        }
        if self.supervise_pose:
            pose_input = {
                "flow_field": flow_out["flow_field"],
                "depth": depth_out["depth"],
                "fl": data["fl"]
            }
            pose_out = self.pose_head(pose_input)
            out["pose"] = pose_out["pose"]
            out["residual"] = pose_out["residual"]

        return out

    @torch.no_grad()
    def infer(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Infer the relative camera motion from the motion-blurred image.

        Args:
            data: Dictionary containing the motion-blurred image and the focal length.

        Returns:
            Dictionary containing the predicted flow field, depth, pose, and residual.
        """
        if type(data["image"]) != torch.Tensor:
            data["image"] = to_tensor(data["image"]).unsqueeze(0)
        data["fx"] = torch.tensor(data["fx"], dtype=torch.float32).unsqueeze(0)
        data["fy"] = torch.tensor(data["fy"], dtype=torch.float32).unsqueeze(0)

        data["fx"] = data["fx"].to(self.device)
        data["fy"] = data["fy"].to(self.device)
        data["image"] = data["image"].to(self.device)

        # Center crop to 320:224 aspect ratio, resize to 320x224, and normalize
        h, w = data["image"].shape[-2:]
        target_ratio = 320 / 224
        if w / h > target_ratio:
            w = int(h * target_ratio)
            data["image"] = center_crop(data["image"], (h, w))
        elif w / h < target_ratio:
            h = int(w / target_ratio)
            data["image"] = center_crop(data["image"], (h, w))
        
        data["image"] = resize(data["image"], size=(224, 320), antialias=True)
        data["image"] = normalize(data["image"], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Adjust focal length for cropping and resizing
        scale_x = 320 / w
        scale_y = 224 / h
        data["fx"] = data["fx"] * scale_x
        data["fy"] = data["fy"] * scale_y
        data["fl"] = (data["fx"] + data["fy"]) / 2
        out =  self(data)

        return out