import torch
import hydra
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from iaai.model import Blur2PoseSegNeXtBackbone
from iaai.dataset import MotionBlurredDataset
from iaai.utils.io import load_ckpt

@hydra.main(config_path="./configs", config_name="gen_pseudo_cfg", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)

    assert cfg.MODEL.POSE_HEAD.SUPERVISE, "Must supervise pose if generating pseudo gt"
    assert cfg.TRAINING.POSE_ONLY, "Must be pose only if generating pseudo gt"

    # Load the dataset
    df = pd.read_csv(cfg.DATASET.CSV_OG_PATH)
    dataset = MotionBlurredDataset(cfg.DATASET.CSV_OG_PATH, cfg, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load the model
    model = Blur2PoseSegNeXtBackbone(supervise_pose=True)
    ckpt = load_ckpt(cfg.CHECKPOINT_PATH)
    model.load_state_dict(ckpt, strict=False)
    model.eval().cuda()

    # Generate the pseudo gt
    flow_dir = Path(cfg.DATASET.LABELS_SAVE_DIR) / "flow"
    depth_dir = Path(cfg.DATASET.LABELS_SAVE_DIR) / "depth"
    flow_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    flow_paths = []
    depth_paths = []
    for i, data in enumerate(dataloader):
        print(f"Processing {i} / {len(dataloader)}")

        img_BCHW = data["blurred"]
        blurred_path = Path(data["blurred_path"][0])
        fl = data["K"][:,0,0]

        with torch.no_grad():
            data = {
                "image": img_BCHW.cuda(),
                "fl": fl.cuda()
            }
            out = model(data)
            flow_B2HW = out["flow_field"]
            depth_BHW = out["depth"]
        # Save the flow and depth
        seq_name = blurred_path.parent.parent.stem
        flow_path = flow_dir / f"{seq_name}_{blurred_path.stem}.pt"
        depth_path = depth_dir / f"{seq_name}_{blurred_path.stem}.pt"
        torch.save(flow_B2HW.squeeze(0).cpu(), str(flow_path))
        torch.save(depth_BHW.squeeze(0).cpu(), str(depth_path))
        flow_paths.append(flow_path)
        depth_paths.append(depth_path)

    # Add the flow and depth columns to the dataframe
    df["flow"] = flow_paths
    df["depth"] = depth_paths
    df.to_csv(cfg.DATASET.CSV_SAVE_PATH, index=False)

if __name__ == "__main__":
    main()