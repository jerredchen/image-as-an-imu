import torch
import imageio
import cv2
import pandas as pd
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from PIL import Image
from omegaconf import ListConfig
from roma import unitquat_to_rotmat

class MotionBlurredDataset(Dataset):
    def __init__(self, csv_file, cfg, is_train=True):
        self.data = pd.read_csv(csv_file)
        self.cfg = cfg
        
        transform = []
        # CenterCrop (ensure the dims are integers)
        if cfg.DATASET.CENTERCROP_DIMS is not None:
            if isinstance(cfg.DATASET.CENTERCROP_DIMS, ListConfig):
                crop_h, crop_w = map(int, cfg.DATASET.CENTERCROP_DIMS)
            else:
                crop_h = crop_w = int(cfg.DATASET.CENTERCROP_DIMS)
            transform.append(A.CenterCrop(
                height=crop_h,
                width=crop_w,
                p=1.0
            ))

        if is_train:
            if "color_transforms" in cfg.DATASET.DATA_AUGMENT:
                color_transforms = [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.4),
                    A.ToGray(p=0.2)
                ]
                transform.extend(color_transforms)

            if "noise_transforms" in cfg.DATASET.DATA_AUGMENT:
                noise_transforms = [
                    A.ImageCompression(quality_range=(20, 100), p=1),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.1), p=0.5),
                    A.Defocus(radius=(1.0, 1.5), alias_blur=(0.01, 0.05), p=0.5),
                ]
                transform.extend(noise_transforms)

        # Add final normalization and conversion to tensor
        transform.extend([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            A.ToTensorV2()
        ])
        
        self.transform = A.Compose(transform)
        self.pose_only = cfg.TRAINING.POSE_ONLY
        self.use_pseudo_gt = cfg.TRAINING.USE_PSEUDO_GT

    def __len__(self):
        # Return the number of data points
        return len(self.data)

    def _load_image(self, path):
        """Helper function to load an image from the file path."""
        return Image.open(path).convert('RGB')

    def _load_depth(self, depth_path):
        """Helper function to load a depth map from the file path."""
        depth = np.asarray(imageio.v2.imread(depth_path)).astype(np.float32)
        depth = torch.from_numpy(depth) / 1000.
        return depth

    def __getitem__(self, idx):
        blurred_image_path = self.data["blurred"][idx]
        if not self.pose_only or self.use_pseudo_gt:
            matches_path = self.data["flow"][idx]
            depth_path = self.data["depth"][idx]
        
        blurred_image_og = cv2.cvtColor(cv2.imread(blurred_image_path), cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=blurred_image_og)
        blurred_image = transformed['image']  # This will be a tensor

        if self.use_pseudo_gt:
            depth_HW = torch.load(depth_path, weights_only=True).squeeze()
            flow_2HW = torch.load(matches_path, weights_only=True)
        if not self.pose_only:
            depth_HW = self._load_depth(depth_path)
            flow_2HW = torch.load(matches_path, weights_only=True)

        # Add center cropping for flow field to match image dimensions
        if self.cfg.DATASET.CENTERCROP_DIMS is not None and not self.pose_only and not self.use_pseudo_gt:
            # Get current flow dimensions
            _, H, W = flow_2HW.shape
            # Calculate crop dimensions
            crop_h, crop_w = self.cfg.DATASET.CENTERCROP_DIMS
            # Calculate starting points for center crop
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            # Apply center crop to flow field
            flow_2HW = flow_2HW[:, start_h:start_h + crop_h, start_w:start_w + crop_w]

            depth_H, depth_W = depth_HW.shape
            start_h = (depth_H - crop_h) // 2
            start_w = (depth_W - crop_w) // 2
            depth_HW = depth_HW[start_h:start_h + crop_h, start_w:start_w + crop_w]

        bRa_qXYZW_4 = torch.tensor(self.data[["bRa_qx", "bRa_qy", "bRa_qz", "bRa_qw"]].iloc[idx].values)
        bRa_33 = unitquat_to_rotmat(bRa_qXYZW_4[None]).squeeze()
        bta_3 = torch.tensor(self.data[["bta_x", "bta_y", "bta_z"]].iloc[idx].values)
        fx_px, fy_px = self.data["fx"][idx], self.data["fy"][idx]
        cx_px, cy_px = self.data["cx"][idx], self.data["cy"][idx]

        if self.cfg.DATASET.CENTERCROP_DIMS is not None:
            # Obtain the resized image dimensions (using PIL (width, height) ordering)
            w_resized, h_resized = np.array(blurred_image_og.shape[:2])[::-1]
            # Calculate the crop offsets
            crop_x = (w_resized - self.cfg.DATASET.CENTERCROP_DIMS[1]) // 2
            crop_y = (h_resized - self.cfg.DATASET.CENTERCROP_DIMS[0]) // 2
            # Adjust principle points by subtracting the crop offset
            cx_px = cx_px - crop_x
            cy_px = cy_px - crop_y

        K_33 = torch.tensor([
            [fx_px, 0, cx_px],
            [0, fy_px, cy_px],
            [0, 0, 1]
        ])

        data = {
            'blurred': blurred_image,
            'blurred_path': blurred_image_path,
            "bRa": bRa_33.float(),
            "bta": bta_3.float(),
            "K": K_33.float(),
        }

        if not self.pose_only or self.use_pseudo_gt:
            data["flow"] = flow_2HW.squeeze()
            data["depth"] = depth_HW.float()
        
        return data
