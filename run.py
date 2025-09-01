import torch
from PIL import Image
from iaai.model import Blur2PoseSegNeXtBackbone
from iaai.funcs import compute_single_image_velocity

model = Blur2PoseSegNeXtBackbone(device="cuda")
state_dict = torch.load("/path/to/model_ckpt.pth")
model.load_state_dict(state_dict, strict=True)
model.eval().cuda()

demo_img = Image.open("./assets/demo.jpg")
data = {"image": demo_img, "fx": 1433.1024, "fy": 1433.1024}
out = model.infer(data) # out is a dict with keys "flow_field", "depth", "pose", "residual"
# Optionally apply a transformation to the rotation matrix
# This transforms the rotation to match the iPhone gyroscope's coordinate system
opencv2gyro_tf = torch.tensor([[ 0, -1,  0],
                               [-1,  0,  0],
                               [ 0,  0, -1]]).float()
rot_vel, trans_vel = compute_single_image_velocity(
    out["pose"].squeeze(),
    exposure_time_s=0.01,
    M=opencv2gyro_tf,
    device="cuda")

# Note that the velocity has sign ambiguity if there is only one image
print(f"Rotational velocity: {rot_vel.cpu().numpy()}")
print(f"Translational velocity: {trans_vel.cpu().numpy()}")