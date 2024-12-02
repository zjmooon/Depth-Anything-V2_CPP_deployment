import torch.onnx 
import onnxruntime as ort

import numpy as np

from onnx import checker
import argparse

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2



# setup
# model = resnet("18", pretrained=True)
## need modify 
parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')

    
parser.add_argument('--img-path', type=str)
parser.add_argument('--input-size', type=int, default=518)
parser.add_argument('--outdir', type=str, default='./vis_depth')

parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--load-from', type=str, default='/home/hud/ssd/Mobility_AR/Depth-Anything-V2/checkpoints/depth_anything_v2_metric_vkitti_vits.pth')
parser.add_argument('--max-depth', type=float, default=80)

parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
args = parser.parse_args()
depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
depth_anything = depth_anything.to(DEVICE).eval()
raw_image = cv2.imread("/home/hud/ssd/Mobility_AR/Depth-Anything-V2/assets/automotive_sample/1920.jpg")


depth_last, image_input, depth_infer = depth_anything.infer_image_export(raw_image, args.input_size)

# 确保所有张量都在同一设备上
image_input = image_input.to(DEVICE)
depth_last = depth_last.to(DEVICE)
depth_infer = depth_infer.to(DEVICE)

# 打印设备信息
print(f"depth_anything device: {next(depth_anything.parameters()).device}")
print(f"image_input device: {image_input.device}")
print(f"depth_last device: {depth_last.device}")
print(f"depth_infer device: {depth_infer.device}")

    
# 检查模型内部所有参数和缓冲区的设备
def check_device_consistency(model):
    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device: {param.device}")
    for name, buffer in model.named_buffers():
        print(f"Buffer {name} is on device: {buffer.device}")

check_device_consistency(depth_anything)

# # model.load_state_dict(torch.load("./Ultra-Fast-Lane-Detection-v2/resnet18-5c106cde.pth"),strict=False)
# if dataset=="culane":
#     x = torch.randn(1, 3, 320, 1600)
#     x_input_new = torch.randn(1, 3, 320, 1600)  ##get the different input from converting process , verification test
#     path = "./Models/culane_res18.pth"
#     out_path = './Models/culane_res18.onnx'
#     model = parsingNet(backbone='18',num_grid_row=200,num_cls_row=72,num_lane_on_row=4,num_grid_col=100,num_cls_col=81,num_lane_on_col=4,input_width=1600,input_height=320)  
# if dataset =="tusimple":
#     x = torch.randn(1, 3, 320, 800)
#     x_input_new = torch.randn(1, 3, 320, 800)  ##get the different input from converting process , verification test
#     path ="./Models/tusimple_res18.pth"
#     out_path = './Models/tusimple_res18.onnx'
#     model = parsingNet(backbone='18',num_grid_row=100,num_cls_row=56,num_lane_on_row=4,num_grid_col=100,num_cls_col=41,num_lane_on_col=4,input_width=800,input_height=320)  

# state_dict = torch.load(path, map_location='cpu')['model']

# ##remove no need string 'model'form name of each layer
# if dataset=="culane":
#     compatible_state_dict = {}
#     for k, v in state_dict.items():
#         if 'module.' in k:
#             compatible_state_dict[k[7:]] = v
#         else:
#             compatible_state_dict[k] = v

#     model.load_state_dict(compatible_state_dict, strict=False)
# if dataset =="tusimple":
#     model.load_state_dict(state_dict, strict=False)
# model.eval()

# PyTorch reference output
# with torch.no_grad():
#     out = model(x_input_new)

# model.eval()
# export to ONNX
onnx_path = 'depth_anything_v2_metric_vkitti_vits_metric_20240725.onnx'
torch.onnx.export(
    depth_anything.to('cpu'),
    image_input.to('cpu'),
    onnx_path,
    export_params=True,  # store the trained parameter weights inside the model file 
    opset_version=11,    # the ONNX version to export the model to 
    input_names = ['input'],   # the model's input names 
    output_names = ['output'], # the model's output names 
)

# ONNX reference output
# ort_session = ort.InferenceSession('culane_res18.onnx',providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'])
ort_session = ort.InferenceSession(onnx_path,providers = ['CUDAExecutionProvider'])
outputs = ort_session.run(
    None,
    {"input": image_input.cpu().numpy()},
)
checker.check_model(onnx_path, True)
# onnx_model = torch.onnx.load('new_onnx/resnet-erasing.onnx')
# torch.onnx.checker.check_model(onnx_model)

# compare ONNX Runtime and PyTorch results

t1 = depth_infer.to('cpu')
print(f"pytorch : {t1.detach().numpy()[:100]}")
print(f"onnx : {outputs[0][:100]}")
print(np.max(np.abs(np.array(t1.detach().numpy()) - outputs[0])))
