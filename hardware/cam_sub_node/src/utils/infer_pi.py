import numpy as np
from torchvision import transforms
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import time
from cityscapesscripts.helpers.labels import trainId2label as t2l
import torch.nn.functional as F
from model.model import SegFormer
import albumentations as A
from albumentations.pytorch import ToTensorV2


if torch.cuda.is_available():
    device = 'cuda:0'
    print("using gpu")
else:
    device = 'cpu'


weight_path = '/home/mys/catkin_ws/src/cam_sub_node/src/utils/log_100.pt'
print('model loaded')
checkpoint = torch.load(weight_path)

model = SegFormer(
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[16, 8, 4, 2],
    num_classes=19,
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),])

colormap = np.zeros((19, 3), dtype=np.uint8)
colormap[0] = [128, 64, 128]
colormap[1] = [244, 35, 232]
colormap[2] = [70, 70, 70]
colormap[3] = [102, 102, 156]
colormap[4] = [190, 153, 153]
colormap[5] = [153, 153, 153]
colormap[6] = [250, 170, 30]
colormap[7] = [220, 220, 0]
colormap[8] = [107, 142, 35]
colormap[9] = [152, 251, 152]
colormap[10] = [70, 130, 180]
colormap[11] = [220, 20, 60]
colormap[12] = [255, 0, 0]
colormap[13] = [0, 0, 142]
colormap[14] = [0, 0, 70]
colormap[15] = [0, 60, 100]
colormap[16] = [0, 80, 100]
colormap[17] = [0, 0, 230]
colormap[18] = [119, 11, 32]

def decode_segmap(pred_labs):
    pred_labs = pred_labs.cpu().numpy()[0]

    r = pred_labs.copy()
    g = pred_labs.copy()
    b = pred_labs.copy()
    for l in range(0, 19):
        r[pred_labs == l] = colormap[l][0]
        g[pred_labs == l] = colormap[l][1]
        b[pred_labs == l] = colormap[l][2]

    rgb = np.zeros((pred_labs.shape[0], pred_labs.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    rgb  = cv2.cvtColor(rgb.astype('float32'), cv2.COLOR_RGB2BGR) 
    return rgb

def get_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(img)
    if transform is not None:
            og_img = np.array(frame)
            transformed = transform(image = og_img)
            image = transformed["image"]
            image = image.unsqueeze(0)
            return image, og_img
    return None, None

def get_labels(predictions):
    predictions = torch.nn.functional.softmax(predictions, dim=1)
    pred_labels = torch.argmax(predictions, dim=1) 
    pred_labels = pred_labels.float()
    return pred_labels
    
def predict_img(model, image, transform):
    with torch.no_grad():
        image = image.to(device)
        predictions = model(image) 
        pred_labels = get_labels(predictions)
        pred_labels = transforms.Resize((64*2, 128*2))(pred_labels)
        color_labels = decode_segmap(pred_labels)
        return color_labels