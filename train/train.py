import time
import logging
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import hydra
import roma
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from iaai.dataset import MotionBlurredDataset
from iaai.model import Blur2PoseSegNeXtBackbone
from iaai.utils.metrics import translation_angle
from iaai.utils.viz import visualize_quivers, visualize_hsv, visualize_depth_maps

logger = logging.getLogger(__name__)

@hydra.main(config_path="./configs", config_name="train_cfg", version_base="1.2")
def train_runner(cfg: DictConfig):
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    assert not cfg.TRAINING.USE_PSEUDO_GT or cfg.TRAINING.POSE_ONLY, "USE_PSEUDO_GT can only be True if POSE_ONLY is also True."

    if cfg.TRAINING.POSE_ONLY and not cfg.TRAINING.USE_PSEUDO_GT:
        cfg.MODEL.POSE_HEAD.SUPERVISE = True
        cfg.TRAINING.POSE_LOSS_WEIGHT = 1.0
        cfg.TRAINING.FLOW_LOSS_WEIGHT = 0.0
        cfg.TRAINING.DEPTH_LOSS_WEIGHT = 0.0
    elif cfg.TRAINING.POSE_ONLY and cfg.TRAINING.USE_PSEUDO_GT:
        cfg.MODEL.POSE_HEAD.SUPERVISE = True

    if cfg.get("DEBUG", False) or cfg.get("debug", False):
        # Load the debug config
        logger.info("DEBUG MODE ENABLED")
        debug_cfg = OmegaConf.load("./configs/debug_cfg.yml")
        cfg = OmegaConf.merge(cfg, debug_cfg)
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    accelerator = Accelerator(
        device_placement=True,
        log_with=cfg.LOGGING.PLATFORM,
        mixed_precision=cfg.TRAINING.MIXED_PRECISION
    )

    # Create run name
    run_name = f"{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
    if cfg.RUN_NAME is not None:
        run_name = f"{cfg.RUN_NAME}_{run_name}"
    accelerator.init_trackers(
        project_name="motion-blur", 
        config={
            # Convert the config to dictionary for logging
            "MODEL": OmegaConf.to_container(cfg.MODEL),
            "TRAINING": OmegaConf.to_container(cfg.TRAINING),
            "DATASET": OmegaConf.to_container(cfg.DATASET),
        },
        init_kwargs={"wandb":{"name":run_name}}
    )

    if accelerator.is_main_process:
        logger.info(f"Run name: {run_name}")
        logger.info(OmegaConf.to_yaml(cfg))

    if cfg.MODEL.TYPE == "mscan":
        model = Blur2PoseSegNeXtBackbone(supervise_pose=cfg.MODEL.POSE_HEAD.SUPERVISE,
                                         use_pinv=cfg.MODEL.POSE_HEAD.USE_PINV)
    else:
        raise ValueError(f"Invalid model type: {cfg.MODEL.TYPE}")

    # Load the backbone weights
    if accelerator.is_main_process:
        logger.info(f"Loading backbone weights from {cfg.MODEL.ENCODER.WEIGHTS}")
    if cfg.MODEL.TYPE == 'mscan':
        state_dict = torch.load(cfg.MODEL.ENCODER.WEIGHTS, weights_only=False)['state_dict']
        model.backbone.load_state_dict(state_dict, strict=False)
    elif cfg.MODEL.TYPE == 'da':
        state_dict = torch.load(cfg.MODEL.ENCODER.WEIGHTS, weights_only=False)
        model.pretrained.load_state_dict(state_dict, strict=False)
    if not cfg.MODEL.POSE_HEAD.SUPERVISE:
        for name, param in model.named_parameters():
            if "pose_head" in name:
                param.requires_grad = False

    train_dataset = MotionBlurredDataset(
        csv_file=cfg.DATASET.TRAIN_DATA,
        cfg=cfg,
        is_train=True
    )
    val_dataset = MotionBlurredDataset(
        csv_file=cfg.DATASET.VAL_DATA,
        cfg=cfg,
        is_train=False
    )

    # Drop the last batch if only single process (accelerator multi-gpu handles noneven batches by default)
    drop_last = accelerator.num_processes == 1
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAINING.NUM_WORKERS,
        drop_last=drop_last
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAINING.NUM_WORKERS,
        drop_last=drop_last
    )

    if accelerator.is_main_process:
        logger.info(f"Total number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
   
    if cfg.TRAINING.OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAINING.LR)
    elif cfg.TRAINING.OPTIMIZER == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR)
    else:
        raise ValueError(f"Invalid optimizer: {cfg.TRAINING.OPTIMIZER}")

    # Create learning rate scheduler
    scheduler = None
    scheduler_type = "none"
    if cfg.TRAINING.get("LR_SCHEDULER", {}).get("TYPE", "none").lower() != "none":
        scheduler_type = cfg.TRAINING.LR_SCHEDULER.TYPE.lower()
        num_epochs = cfg.TRAINING.EPOCHS
        steps_per_epoch = len(train_dataloader)
        total_steps = num_epochs * steps_per_epoch
        
        if scheduler_type == "onecycle":
            if accelerator.is_main_process:
                logger.info(f"Using OneCycleLR scheduler with max_lr={cfg.TRAINING.LR_SCHEDULER.MAX_LR}")
            # OneCycleLR parameters - only cycle momentum if using SGD
            cycle_momentum = cfg.TRAINING.LR_SCHEDULER.get("CYCLE_MOMENTUM", True) and isinstance(optimizer, optim.SGD)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=cfg.TRAINING.LR_SCHEDULER.get("MAX_LR", 0.001),
                total_steps=total_steps,
                pct_start=cfg.TRAINING.LR_SCHEDULER.get("PCT_START", 0.3),
                anneal_strategy=cfg.TRAINING.LR_SCHEDULER.get("ANNEAL_STRATEGY", "cos"),
                cycle_momentum=cycle_momentum,
                base_momentum=cfg.TRAINING.LR_SCHEDULER.get("BASE_MOMENTUM", 0.85) if cycle_momentum else None,
                max_momentum=cfg.TRAINING.LR_SCHEDULER.get("MAX_MOMENTUM", 0.95) if cycle_momentum else None,
                div_factor=cfg.TRAINING.LR_SCHEDULER.get("DIV_FACTOR", 25.0),
                final_div_factor=cfg.TRAINING.LR_SCHEDULER.get("FINAL_DIV_FACTOR", 10000.0),
                three_phase=cfg.TRAINING.LR_SCHEDULER.get("THREE_PHASE", False),
            )
        else:
            if accelerator.is_main_process:
                logger.warning(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        val_dataloader
    )
    
    # Prepare scheduler with accelerator if it exists
    if scheduler is not None:
        scheduler = accelerator.prepare(scheduler)

    num_epochs = cfg.TRAINING.EPOCHS
    model = model.to(accelerator.device)

    if cfg.TRAINING.CHECKPOINT.RESUME_PATH:
        # Load only the model weights, not the optimizer state
        checkpoint_path = Path(cfg.TRAINING.CHECKPOINT.RESUME_PATH)
        model_state_path = checkpoint_path / "model.safetensors"
        
        if model_state_path.exists():
            if accelerator.is_main_process:
                logger.info(f"Loading model weights from {model_state_path}")
            
            # Load the state dict from safetensors and apply it to the unwrapped model
            from safetensors.torch import load_file
            state_dict = load_file(model_state_path)
            accelerator.unwrap_model(model).load_state_dict(state_dict, strict=False)
        else:
            if accelerator.is_main_process:
                logger.warning(f"Model checkpoint not found at {model_state_path}")
    # Create checkpoint folder with run name
    checkpoint_run_path = Path(cfg.TRAINING.CHECKPOINT.DIR) / run_name
    if accelerator.is_main_process and cfg.TRAINING.CHECKPOINT.SAVE:
        checkpoint_run_path.mkdir(exist_ok=True, parents=True)
        if accelerator.is_main_process:
            logger.info(f"Saving checkpoints to {str(checkpoint_run_path)}")
    accelerator.wait_for_everyone()

    try:
        for epoch in range(num_epochs):
            accelerator.print(f"****************Epoch {epoch+1}/{num_epochs}****************")
            if ((cfg.TRAINING.VAL_FIRST and epoch == 0) or (epoch != 0 and epoch % cfg.TRAINING.VAL_INTERVAL == 0)):
                    if accelerator.is_main_process:
                        logger.info("RUNNING VALIDATION...")
                    with torch.no_grad():
                        results = run(
                            val_dataloader,
                            model,
                            optimizer,
                            accelerator,
                            cfg,
                            train=False,
                            scheduler=scheduler,
                            STEPS_PER_PRINT=cfg.LOGGING.STEPS_PER_PRINT)
                    accelerator.log({
                        "val/loss": results["loss"],
                        "epoch": epoch
                    })
                    if not cfg.TRAINING.POSE_ONLY:
                        accelerator.log({
                            "val/epe": results["epe"],
                            "val/absrel_depth": results["absrel_depth"],
                            "epoch": epoch
                        })
                    if cfg.MODEL.POSE_HEAD.SUPERVISE:
                        accelerator.log({
                            "val/rot_angle_error": results["rot_angle_error"],
                            "val/trans_angle_error": results["trans_angle_error"],
                            "val/trans_dist_error": results["trans_dist_error"],
                            "val/pose_error": (10*results["rot_angle_error"] + results["trans_angle_error"])/2,
                            "epoch": epoch
                        })
                    wandb_quivers = wandb.Image(results["quivers"])
                    wandb_hsv = wandb.Image(results["hsv"])
                    wandb_depth_maps = wandb.Image(results["depth_maps"])
                    accelerator.log({
                        "val/quivers": [wandb_quivers],
                        "val/hsv": [wandb_hsv],
                        "val/depth_maps": [wandb_depth_maps],
                        "epoch": epoch
                    })
                    del wandb_quivers, wandb_hsv, wandb_depth_maps
            if accelerator.is_main_process:
                logger.info("RUNNING TRAINING...")
            results = run(
                train_dataloader,
                model,
                optimizer,
                accelerator,
                cfg,
                train=True,
                scheduler=scheduler,
                STEPS_PER_PRINT=cfg.LOGGING.STEPS_PER_PRINT
            )
            current_lr = optimizer.param_groups[0]['lr']
            accelerator.log({
                "train/loss": results["loss"],  
                "train/learning_rate": current_lr,
                "epoch": epoch
            })
            if not cfg.TRAINING.POSE_ONLY:
                accelerator.log({
                    "train/flow_loss": results["flow_loss"],
                    "train/depth_loss": results["depth_loss"],
                    "train/epe": results["epe"],
                    "epoch": epoch
                })
            if cfg.MODEL.POSE_HEAD.SUPERVISE:
                accelerator.log({
                    "train/pose_loss": results["pose_loss"],
                    "train/rot_angle_error": results["rot_angle_error"],
                    "train/trans_angle_error": results["trans_angle_error"],
                    "train/trans_dist_error": results["trans_dist_error"],
                    "epoch": epoch
                })

            keys = list(results.keys())
            for key in keys:
                del results[key]
            del results, keys
            
            # Step scheduler if it exists and scheduler_type is not onecycle (OneCycleLR steps per batch, not per epoch)
            if scheduler is not None and scheduler_type != "onecycle":
                scheduler.step()
                
            if cfg.TRAINING.CHECKPOINT.SAVE and epoch % cfg.TRAINING.CHECKPOINT.INTERVAL == 0:
                accelerator.wait_for_everyone()
                if cfg.TRAINING.CHECKPOINT.OVERWRITE:
                    ckpt_epoch_path = checkpoint_run_path / "ckpt_latest"
                else:
                    ckpt_epoch_path = checkpoint_run_path / f"ckpt_{epoch:06}"
                if accelerator.is_main_process:
                    logger.info(f"----------Saving the ckpt at epoch {epoch} to {ckpt_epoch_path}----------")
                accelerator.save_state(output_dir=ckpt_epoch_path)
                if cfg.LOGGING.PLATFORM == "wandb" and accelerator.is_main_process:
                    wandb.save(f"{ckpt_epoch_path}/*", base_path=checkpoint_run_path)
        if cfg.TRAINING.CHECKPOINT.SAVE:
            # Perform a final checkpoint save at the end of training
            ckpt_epoch_path = checkpoint_run_path / f"ckpt_{epoch:06}"
            accelerator.save_state(output_dir=ckpt_epoch_path)
            if cfg.LOGGING.PLATFORM == "wandb" and accelerator.is_main_process:
                wandb.save(f"{ckpt_epoch_path}/*", base_path=checkpoint_run_path)
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logger.info("Keyboard interrupt detected.")
        if cfg.TRAINING.CHECKPOINT.SAVE:
            if accelerator.is_main_process:
                logger.info("Saving model and exiting...")
            ckpt_epoch_path = checkpoint_run_path / f"ckpt_{epoch:06}"
            accelerator.save_state(output_dir=ckpt_epoch_path)
            if cfg.LOGGING.PLATFORM == "wandb" and accelerator.is_main_process:
                wandb.save(f"{ckpt_epoch_path}/*", base_path=checkpoint_run_path)
        else:
            if accelerator.is_main_process:
                logger.info("Exiting...")
        exit()

def run(dataloader, model, optimizer, accelerator, cfg, train=True, scheduler=None, STEPS_PER_PRINT=10):
    if train:
        model.train()
    else:
        model.eval()
    running_loss = 0.0
    running_pose_loss = running_flow_loss = running_depth_loss = 0.0
    running_epe = running_absrel_depth = 0.0
    running_rangle_deg = running_tangle_deg = running_terror_m = 0.0
    torch.cuda.synchronize()
    start_time = time.time()
    for i, data in enumerate(dataloader):

        # Get the inputs and ground truth keypoints
        blurred_img_BCHW = data['blurred'].to(accelerator.device)
        gt_bRa_B33 = data['bRa'].to(accelerator.device)
        gt_bta_B3 = data['bta'].to(accelerator.device)
        gt_K_B33 = data['K'].to(accelerator.device)
        gt_fx_B = gt_K_B33[...,0,0]
        gt_fy_B = gt_K_B33[...,1,1]
        gt_fl_B = torch.mean(torch.stack([gt_fx_B, gt_fy_B], dim=-1), dim=-1)
        B, _, h, w = blurred_img_BCHW.shape
        if not cfg.TRAINING.POSE_ONLY or cfg.TRAINING.USE_PSEUDO_GT:
            gt_flow_B2HW = data['flow'].to(accelerator.device)
            gt_depth_BHW = data['depth'].to(accelerator.device)
        else:
            gt_flow_B2HW = torch.zeros(B, 2, h, w)
            gt_depth_BHW = torch.zeros(B, h, w)

        # Forward pass
        data = {
            "image": blurred_img_BCHW,
            "fl": gt_fl_B
        }
        output = model(data)
        pred_pose_B6 = output["pose"]
        pred_residual_B6 = output["residual"]
        pred_flow_B2HW = output["flow_field"]
        pred_depth_BHW = output["depth"].squeeze()

        ######################## LOSS ########################

        # Compute pose loss
        if cfg.MODEL.POSE_HEAD.SUPERVISE:
            pred_aRb_B33 = roma.euler_to_rotmat('XYZ', pred_pose_B6[:, :3].squeeze())
            pred_atb_B3 = pred_pose_B6[:, 3:].squeeze()
            pred_aRb_B33_inv = pred_aRb_B33.transpose(1, 2)
            pred_atb_B3_inv = (-pred_aRb_B33_inv @ pred_atb_B3.unsqueeze(-1)).squeeze()
            gt_aRb_B33 = gt_bRa_B33.transpose(1, 2)
            gt_atb_B3 = -(gt_aRb_B33 @ gt_bta_B3.unsqueeze(-1)).squeeze()

            # Compute if pred is closer to gt_aTb vs gt_bTa
            pred_aTb_vec_B6 = torch.cat([roma.rotmat_to_rotvec(pred_aRb_B33[None]).squeeze(), pred_atb_B3], dim=1)
            gt_aTb_vec_B6 = torch.cat([roma.rotmat_to_rotvec(gt_aRb_B33[None]).squeeze(), gt_atb_B3], dim=1)
            gt_bTa_vec_B6 = torch.cat([roma.rotmat_to_rotvec(gt_bRa_B33[None]).squeeze(), gt_bta_B3], dim=1)
            dot_prod_aTb_B1 = torch.sum(pred_aTb_vec_B6 * gt_aTb_vec_B6, dim=1)
            dot_prod_bTa_B1 = torch.sum(pred_aTb_vec_B6 * gt_bTa_vec_B6, dim=1)
            reorient_mask_B = (dot_prod_aTb_B1 > dot_prod_bTa_B1).bool()
            gt_reorient_rots = torch.where(reorient_mask_B[:,None,None].expand(-1, 3, 3), gt_aRb_B33, gt_bRa_B33)
            gt_reorient_trans = torch.where(reorient_mask_B[:,None].expand(-1, 3), gt_atb_B3, gt_bta_B3)
            pose_loss = cfg.TRAINING.ROT_LOSS_WEIGHT * F.mse_loss(pred_aRb_B33, gt_reorient_rots) + cfg.TRAINING.TRANS_LOSS_WEIGHT * F.mse_loss(pred_atb_B3, gt_reorient_trans)
        else:
            pose_loss = torch.tensor(0.0).to(accelerator.device)

        if not cfg.TRAINING.POSE_ONLY:
            # Compute flow loss
            # Determine which gt flow direction has smallest dot product with pred flow
            gt_flow_fw_B2HW = gt_flow_B2HW.clone()
            gt_flow_bw_B2HW = -gt_flow_B2HW 
            dot_prod_fw_B1HW = torch.sum(pred_flow_B2HW * gt_flow_fw_B2HW, dim=1, keepdim=True)
            dot_prod_bw_B1HW = torch.sum(pred_flow_B2HW * gt_flow_bw_B2HW, dim=1, keepdim=True)
            reorient_mask = (dot_prod_fw_B1HW.sum(dim=(2,3)) > dot_prod_bw_B1HW.sum(dim=(2,3))).bool()
            gt_flow_reoriented_B2HW = torch.where(reorient_mask[...,None,None].expand(-1, 2, h, w),
                                                    gt_flow_fw_B2HW,
                                                    gt_flow_bw_B2HW)
            flow_loss_BHW = torch.sum(torch.abs(pred_flow_B2HW - gt_flow_reoriented_B2HW), dim=1)
            weighted_flow_loss = flow_loss_BHW.mean()

            # Compute depth loss
            depth_loss_BHW = torch.abs(pred_depth_BHW - gt_depth_BHW)
            weighted_depth_loss = depth_loss_BHW.mean()

            total_loss = cfg.TRAINING.POSE_LOSS_WEIGHT * pose_loss \
                        + cfg.TRAINING.FLOW_LOSS_WEIGHT * weighted_flow_loss \
                        + cfg.TRAINING.DEPTH_LOSS_WEIGHT * weighted_depth_loss
        elif cfg.TRAINING.USE_PSEUDO_GT:
            # Prevent flows and depths from deviating too much from original predictions
            pseudo_gt_flow_B2HW = gt_flow_B2HW.clone().detach()
            pseudo_gt_depth_BHW = gt_depth_BHW.clone().detach()
            weighted_flow_loss = torch.abs(pred_flow_B2HW - pseudo_gt_flow_B2HW).mean()
            weighted_depth_loss = torch.abs(pred_depth_BHW - pseudo_gt_depth_BHW).mean()
            total_loss = pose_loss \
                        + cfg.TRAINING.FLOW_LOSS_WEIGHT * weighted_flow_loss \
                        + cfg.TRAINING.DEPTH_LOSS_WEIGHT * weighted_depth_loss
        else:
            total_loss = pose_loss
            weighted_flow_loss = torch.tensor(0.0)
            weighted_depth_loss = torch.tensor(0.0)

        ######################## METRICS ########################

        if not cfg.TRAINING.POSE_ONLY:
            epe = torch.norm(pred_flow_B2HW - gt_flow_reoriented_B2HW, dim=1).mean().item()

            x = torch.arange(w, device=accelerator.device).float()
            y = torch.arange(h, device=accelerator.device).float()
            y_grid, x_grid = torch.meshgrid(y, x, indexing='xy')
            meshgrid_BN2 = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=-1).unsqueeze(0).repeat(B, 1, 1)
            pred_flow_BN2 = pred_flow_B2HW.permute(0, 2, 3, 1).view(B, -1, 2)
            pred_matches_BN4 = torch.cat([meshgrid_BN2, meshgrid_BN2 + pred_flow_BN2], dim=2)
            pred_flipped_BN4 = torch.cat([meshgrid_BN2 + pred_flow_BN2, meshgrid_BN2], dim=2)

            # Compute AbsRel depth error
            absrel_depth = torch.abs(pred_depth_BHW - gt_depth_BHW) / gt_depth_BHW
            absrel_depth = absrel_depth.mean()
        else:
            epe = 0.0
            absrel_depth = torch.tensor(0.0)

        # Compute the rotational and translational error with both prediction directions
        if cfg.MODEL.POSE_HEAD.SUPERVISE:
            rangle1_deg = roma.rotmat_geodesic_distance(gt_aRb_B33, pred_aRb_B33).mean().item() * 180.0 / np.pi
            rangle2_deg = roma.rotmat_geodesic_distance(gt_aRb_B33, pred_aRb_B33_inv).mean().item() * 180.0 / np.pi
            tangle1_deg = translation_angle(gt_atb_B3, pred_atb_B3, batch_size=B).mean().item()
            tangle2_deg = translation_angle(gt_atb_B3, pred_atb_B3_inv, batch_size=B).mean().item()
            terror1_m = torch.norm(gt_atb_B3 - pred_atb_B3, dim=1).mean().item()
            terror2_m = torch.norm(gt_atb_B3 - pred_atb_B3_inv, dim=1).mean().item()
            if tangle2_deg**2 + rangle2_deg**2 < tangle1_deg**2 + rangle1_deg**2:
                rangle_deg = rangle2_deg
                tangle_deg = tangle2_deg
                pred_aRb_B33 = pred_aRb_B33_inv
                pred_atb_B3 = pred_atb_B3_inv
                terror_m = terror2_m
            else:
                rangle_deg = rangle1_deg
                tangle_deg = tangle1_deg
                terror_m = terror1_m
        else:
            rangle_deg = 0.0
            tangle_deg = 0.0
            terror_m = 0.0

        if train:
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            # Step OneCycleLR scheduler after each batch
            if scheduler is not None and cfg.TRAINING.LR_SCHEDULER.TYPE == "onecycle":
                scheduler.step()

        running_loss = (running_loss*i + total_loss.item()) / (i + 1)
        running_pose_loss = (running_pose_loss*i + pose_loss.item()) / (i + 1)
        running_flow_loss = (running_flow_loss*i + weighted_flow_loss.item()) / (i + 1)
        running_depth_loss = (running_depth_loss*i + weighted_depth_loss.item()) / (i + 1)
        running_absrel_depth = (running_absrel_depth*i + absrel_depth.item()) / (i + 1)
        running_epe = (running_epe*i + epe) / (i + 1)
        running_rangle_deg = (running_rangle_deg*i + rangle_deg) / (i + 1)
        running_tangle_deg = (running_tangle_deg*i + tangle_deg) / (i + 1)
        running_terror_m = (running_terror_m*i + terror_m) / (i + 1)

        if i % STEPS_PER_PRINT == 0:
            torch.cuda.synchronize()
            end_time = time.time()
            sec_per_iter = (end_time - start_time) / (i + 1)
            current_lr = optimizer.param_groups[0]['lr']
            if train and accelerator.is_main_process:
                logger.info(f"Batch {i} | LR: {current_lr:.2e} | Pose Loss: {running_pose_loss:0.6f} | Flow Loss: {running_flow_loss:0.6f} | Depth Loss: {running_depth_loss:0.6f} | EPE: {running_epe:0.6f} | AbsRel Depth: {running_absrel_depth:0.6f} | Rot Error: {running_rangle_deg:0.6f} | Trans Error: {running_tangle_deg:0.6f} | sec/iter: {sec_per_iter:0.3f} s")
            elif not train and accelerator.is_main_process:
                logger.info(f"Batch {i} | EPE: {running_epe:0.6f} | AbsRel Depth: {running_absrel_depth:0.6f} | Rot Error: {running_rangle_deg:0.6f} | Trans Error: {running_tangle_deg:0.6f} | sec/iter: {sec_per_iter:0.3f} s")

        # Only log ongoing loss if training
        if train:
            accelerator.log({"train/loss": total_loss.item()})
            accelerator.log({"train/lr": current_lr})
            if not cfg.TRAINING.POSE_ONLY:
                accelerator.log({"train/flow_loss": weighted_flow_loss.item()})
                accelerator.log({"train/depth_loss": weighted_depth_loss.item()})
            if cfg.MODEL.POSE_HEAD.SUPERVISE:
                accelerator.log({"train/pose_loss": pose_loss.item()})
            
            if i % cfg.LOGGING.TRAIN_VISUALS_INTERVAL == 0:
                # Produce match visualizations for the last batch
                hsv = visualize_hsv(gt_flow_B2HW, pred_flow_B2HW, upscale_term=1, num_images=8, mod_half=True)
                depth_maps = visualize_depth_maps(gt_depth_BHW, pred_depth_BHW.detach(), blurred_img_BCHW, num_images=8)
                quivers = visualize_quivers(blurred_img_BCHW, gt_flow_B2HW, pred_flow_B2HW.detach(), num_images=8)

                wandb_hsv = wandb.Image(hsv)
                wandb_depth_maps = wandb.Image(depth_maps)
                wandb_quivers = wandb.Image(quivers)
                accelerator.log({
                    "train/hsv": [wandb_hsv],
                    "train/depth_maps": [wandb_depth_maps],
                    "train/quivers": [wandb_quivers]
                })
                del hsv, depth_maps, quivers
                del wandb_hsv, wandb_depth_maps, wandb_quivers

        del gt_K_B33
        if not cfg.TRAINING.POSE_ONLY:
            del h, w, x, y, x_grid, y_grid, meshgrid_BN2, pred_flow_BN2, pred_matches_BN4, pred_flipped_BN4, gt_bRa_B33, gt_bta_B3
        if cfg.MODEL.POSE_HEAD.SUPERVISE:
            del gt_aRb_B33, pred_aRb_B33, pred_aRb_B33_inv, pred_atb_B3, pred_atb_B3_inv, gt_atb_B3
            del rangle_deg, tangle_deg, terror_m, rangle1_deg, rangle2_deg, tangle1_deg, tangle2_deg, terror1_m, terror2_m
        if i < len(dataloader) - 1:
            del blurred_img_BCHW, gt_flow_B2HW, pred_flow_B2HW, gt_depth_BHW, pred_depth_BHW

    quivers = None
    hsv = None
    depth_maps = None
    if not train:
        # Produce match visualizations for the last batch
        hsv = visualize_hsv(gt_flow_B2HW, pred_flow_B2HW, upscale_term=1, num_images=8, mod_half=True)
        depth_maps = visualize_depth_maps(gt_depth_BHW, pred_depth_BHW, blurred_img_BCHW, num_images=8)
        quivers = visualize_quivers(blurred_img_BCHW, gt_flow_B2HW, pred_flow_B2HW, num_images=8, subsample_factor=14)

    del blurred_img_BCHW, gt_flow_B2HW, pred_flow_B2HW, gt_depth_BHW, pred_depth_BHW
    if not cfg.TRAINING.POSE_ONLY:
        del gt_flow_reoriented_B2HW

    return {
        "loss": running_loss,
        "pose_loss": running_pose_loss,
        "flow_loss": running_flow_loss,
        "depth_loss": running_depth_loss,
        "epe": running_epe,
        "absrel_depth": running_absrel_depth,
        "rot_angle_error": running_rangle_deg,
        "trans_angle_error": running_tangle_deg,
        "trans_dist_error": running_terror_m,
        "quivers": quivers,
        "hsv": hsv,
        "depth_maps": depth_maps,
    }

if __name__ == "__main__":
    train_runner()
