import os
import sys
import time
import cv2
import torch
import hydra
import roma
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from torchvision.utils import save_image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'ECCV2022-RIFE')))
from train_log.RIFE_HDv3 import Model as RIFEModel
from promptda.promptda import PromptDA

import synth_data_utils as utils
from iaai.utils.io import read_img_path

def process_sequence(cfg, sequence_path, rife_model, pda_model, split, min_px_threshold=0.0, max_px_threshold=30.0):
    # Get all images in the sequence
    seq = sequence_path.stem
    rgb_path = sequence_path / "iphone" / "rgb"
    arkit_depth_path = sequence_path / "iphone" / "depth"
    cam_path = sequence_path / "iphone" / "colmap" / "cameras.txt"
    colmap_imgs_path = sequence_path / "iphone" / "colmap" / "images.txt"
    pose_intr_imu_path = sequence_path / "iphone" / "pose_intrinsic_imu.json"

    if not arkit_depth_path.exists() \
        or len(list(arkit_depth_path.iterdir())) == 0 \
        or not colmap_imgs_path.exists() \
        or not rgb_path.exists() \
        or not pose_intr_imu_path.exists():
        print(f"\033[91mSkipping {seq} because it does not have all the required files\033[0m")
        return []

    cam_data = utils.read_cameras_text(cam_path)
    colmap_imgs = utils.read_images_text(colmap_imgs_path)
    pose_intr_imu = utils.read_arkit_json(pose_intr_imu_path)

    blurred_base_path = Path(cfg.SYNTH_DATA_PATH) / split / "blurred"
    flow_base_path = Path(cfg.SYNTH_DATA_PATH) / split / "flow"
    depth_base_path = Path(cfg.SYNTH_DATA_PATH) / split / "depth"
    blurred_base_path.mkdir(parents=True, exist_ok=True)
    flow_base_path.mkdir(parents=True, exist_ok=True)
    depth_base_path.mkdir(parents=True, exist_ok=True)
    frame_num_gap = cfg.FRAME_NUM_SKIP
    results = []

    colmap_img_keys = list(colmap_imgs.keys())
    colmap_img_keys.sort()
    for i, img_id in enumerate(colmap_img_keys):
        print(f"\033[95m-> Blurring frame {i+1}/{len(colmap_img_keys)}\033[0m")
        try:
            curr_img = colmap_imgs[img_id]
            img1_filename_jpg = Path(curr_img.name).name
            img1_stem = Path(curr_img.name).stem
            img1_frame_num = int(img1_stem.split('_')[-1])

            # Check for any jumps in the ARKit poses
            if img1_frame_num >= 80:
                # Uses handcrafted heuristics to detect jumps in localization
                jump_detected = utils.detect_arkit_pose_jump(pose_intr_imu, img1_stem, frame_num_gap)
                if jump_detected:
                    print(f"\033[91mJump detected for {img1_filename_jpg} from {seq}\033[0m")
                    torch.cuda.empty_cache()
                    continue

            img2_frame_num = img1_frame_num + frame_num_gap
            img2_stem = "frame_" + str(img2_frame_num).zfill(6)
            img2_filename_jpg = img2_stem + '.jpg'
            if img1_filename_jpg == img2_filename_jpg:
                print(f"\033[91mSkipping {img1_filename_jpg} from {seq} because it is the same frame\033[0m")
                torch.cuda.empty_cache()
                continue
            if not (rgb_path / img2_filename_jpg).exists():
                print(f"\033[91mSkipping {img1_filename_jpg} from {seq} because it does not have a corresponding frame\033[0m")
                torch.cuda.empty_cache()
                continue

            # Compute the relative pose 
            wTa_arkit = torch.tensor(pose_intr_imu[img1_stem]['aligned_pose'])
            wTb_arkit = torch.tensor(pose_intr_imu[img2_stem]['aligned_pose'])
            bTa_arkit = utils.invert_pose(wTb_arkit) @ wTa_arkit

            img1_BCHW = read_img_path(rgb_path / img1_filename_jpg, resize=True, data_width=cfg.DATA.WIDTH, data_height=cfg.DATA.HEIGHT)
            img2_BCHW = read_img_path(rgb_path / img2_filename_jpg, resize=True, data_width=cfg.DATA.WIDTH, data_height=cfg.DATA.HEIGHT)
            # Get the camera intrinsic parameters
            K = utils.read_camera_params(curr_img.camera_id, cam_data, cfg.DATA.WIDTH, cfg.DATA.HEIGHT)

            # Get forward flows
            flow_2HW, depth_map_HW = utils.compute_optical_flow(
                img1_BCHW,
                K,
                bTa_arkit,
                arkit_depth_path / img1_filename_jpg.replace('jpg', 'png'),
                pda_model,
                cfg.DATA.HEIGHT,
                cfg.DATA.WIDTH
            )
            px_dist_HW = torch.norm(flow_2HW, dim=0)
            avg_dist_px = px_dist_HW.mean().item()
            max_dist_px = px_dist_HW.max().item()
            print(f"Average Pixel Distance: {avg_dist_px:.2f}")
            if avg_dist_px < min_px_threshold or avg_dist_px > max_px_threshold:
                print(f"\033[91mPixel distance outside threshold for {img1_filename_jpg} from {seq}\033[0m")
                del img1_BCHW, img2_BCHW, K, flow_2HW, depth_map_HW, px_dist_HW
                torch.cuda.empty_cache()
                continue
            blurred_save_path = blurred_base_path / f"{seq}_{i}.png"
            flow_save_path = flow_base_path / f"{seq}_{i}.pt"
            depth_save_path = depth_base_path / f"{seq}_{i}.png"

            # Interpolate in between the two frames using RIFE
            final_frames = utils.interpolate_frames(
                img1_BCHW,
                img2_BCHW,
                img1_frame_num,
                rife_model,
                frame_num_gap,
                rgb_path,
                cfg.DATA.HEIGHT,
                cfg.DATA.WIDTH
            )

            # Convert from gamma space to linear space
            final_frames_linear = final_frames ** 2.2
            blurred_img_CHW = final_frames_linear.mean(dim=0)
            # Convert from linear space to gamma space
            blurred_img_CHW = blurred_img_CHW ** (1/2.2)

            save_image(blurred_img_CHW, blurred_save_path)
            torch.save(flow_2HW, flow_save_path)
            utils.save_depth(depth_map_HW[None, None], depth_save_path)
            bRa_qvec = roma.rotmat_to_unitquat(bTa_arkit[:3, :3].unsqueeze(0)).squeeze()
            bta = bTa_arkit[:3, 3].squeeze()
            results.append({
                'blurred': blurred_save_path,
                'flow': flow_save_path,
                'depth': depth_save_path,
                'image1': str(rgb_path / img1_filename_jpg),
                'image2': str(rgb_path / img2_filename_jpg),
                'ts1': pose_intr_imu[img1_stem]['timestamp'],
                'ts2': pose_intr_imu[img2_stem]['timestamp'],
                'fx': K[0, 0].item(),
                'fy': K[1, 1].item(),
                'cx': K[0, 2].item(),
                'cy': K[1, 2].item(),
                'bRa_qx': bRa_qvec[0].item(),
                'bRa_qy': bRa_qvec[1].item(),
                'bRa_qz': bRa_qvec[2].item(),
                'bRa_qw': bRa_qvec[3].item(),
                'bta_x': bta[0].item(),
                'bta_y': bta[1].item(),
                'bta_z': bta[2].item(),
                'avg_px_dist': avg_dist_px,
                'max_px_dist': max_dist_px
            })
            del img1_BCHW, img2_BCHW, K, flow_2HW, depth_map_HW, px_dist_HW
            del final_frames, final_frames_linear, blurred_img_CHW, bRa_qvec, bta
            torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print("Exiting due to keyboard interrupt.")
            sys.exit(0)
        except Exception as e:
            print(f"\033[91mError processing {seq}, {colmap_imgs[img_id].name}\033[0m")
            with open(f'failures_scannetpp_{split}.txt', 'a') as f:
                f.write(f"{seq},{colmap_imgs[img_id].name},{e}\n")

    return results

@hydra.main(config_path=".", config_name="synth_data_gen_cfg", version_base="1.2")
def main(cfg):
    torch.manual_seed(0)
    splits = cfg.SPLITS
    print(OmegaConf.to_yaml(cfg))

    # Load the RIFE model
    rife_model = RIFEModel()
    rife_model.load_model(cfg.RIFE_MODEL_PATH, -1)
    rife_model.eval()
    rife_model.device()

    # Load the MDE model
    pda_model = PromptDA.from_pretrained("depth-anything/promptda_vitl").to("cuda").eval()

    VAL_SEQS = utils.VAL_SEQS
    all_seq_times = []
    start_time = time.time()
    for split in splits:
        all_results = []
        assert split in ["train", "val"]
        if OmegaConf.select(cfg, f"LOAD_DATASET.{split.upper()}") is not None:
            loaded_path = OmegaConf.select(cfg, f"LOAD_DATASET.{split.upper()}")
            if loaded_path.split('.')[-1] == 'csv':
                df = pd.read_csv(loaded_path)
            elif loaded_path.split('.')[-1] == 'pkl':
                df = pd.read_pickle(loaded_path)
            else:
                raise ValueError("Invalid file format for loaded dataset.")
            all_results.extend(df.to_dict(orient='records'))
        csv_name = f"synth_data_{split}-skip_{cfg.FRAME_NUM_SKIP}.csv"
        pkl_name = f"synth_data_{split}-skip_{cfg.FRAME_NUM_SKIP}.pkl"
        if split == "train":
            sequences = sorted(Path(cfg.SCANNETPP_DATA_PATH).iterdir())
            # Remove the sequences that are in the validation set
            sequences = [seq for seq in sequences if seq not in VAL_SEQS]
        elif split == "val":
            sequences = VAL_SEQS
        else:
            raise ValueError(f"Invalid split: {split}")
        for i, sequence in enumerate(sequences[cfg.START_ITER:]):
            print('-'*20)
            seq_start_time = time.time()
            print(f"Processing sequence: {sequence}")
            sequence_path = Path(cfg.SCANNETPP_DATA_PATH) / sequence
            results = process_sequence(
                cfg,
                sequence_path,
                rife_model,
                pda_model,
                split
            )
            all_results.extend(results)
            seq_time = time.time() - seq_start_time
            all_seq_times.append(seq_time)
            print(f"\033[92mAverage time per sequence: {np.mean(all_seq_times):.6f} seconds\033[0m")
            # Save the DataFrame every sampled number of sequences
            if i % 2 == 0:
                df = pd.DataFrame(all_results)
                df.to_pickle(pkl_name)
        # Create and save the DataFrame
        df = pd.DataFrame(all_results)
        df.to_csv(csv_name, index=False)

    print(f"Total time taken: {(time.time() - start_time) / 3600 :.4f} hours")

if __name__ == '__main__':
    main()