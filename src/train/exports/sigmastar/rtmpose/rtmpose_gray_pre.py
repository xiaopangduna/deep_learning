# import numpy as np
# import cv2
#
# def image_preprocess(image_file, norm=True):
#     im = cv2.imread(image_file)
#     im = cv2.resize(im,(224,224)
#
#     im = im[:, :, ::-1]  # BGR to RGB
#     im = np.expand_dims(im, 0)
#     im = np.ascontiguousarray(im)
#     # im = torch.from_numpy(im)
#     if norm:
#         #return (im.astype(np.float32) - 127.5) / 127.5
#         return im.astype(np.float32) / 255
#     else:
#         return im
import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision

def sim_preprocess(image_file, input_size=256, norm=True):
    # Keep preprocessing channel count aligned with GRAY model input.
    img_ori = Image.open(image_file).convert('L')
    # resize
    # img_width, img_height = img_ori.size
    # scale = input_size / max(img_height, img_width)
    # resized_width, resized_height = int(round(scale * img_width)), int(round(scale * img_height))
    # img_resized = transforms.Resize(
    #     size=(resized_height, resized_width), interpolation=Image.BILINEAR)(img_ori)
    # # pad
    # pad_left = (input_size - resized_width) // 2
    # pad_right = input_size - resized_width - pad_left
    # pad_top = (input_size - resized_height) // 2
    # pad_bot = input_size - resized_height - pad_top
    # img_padded = transforms.Pad(
    #     padding=(pad_left, pad_top, pad_right, pad_bot), fill=0, padding_mode='constant')(img_resized)
    # normalize
    if norm:
        image_np = np.array(img_ori, dtype=np.float32)
        image_np = np.expand_dims(image_np, axis=-1)  # HWC, single channel
        image_np = (image_np - 107.58) / 57.94
        img_normalized = transforms.ToTensor()(image_np)  # HWC -> CHW tensor
        image = torch.unsqueeze(img_normalized, 0)  # NCHW
        image = np.transpose(np.array(image), (0, 2, 3, 1)).astype(np.float32)
    else:
        image = np.array(img_ori)
        image = np.expand_dims(image, 0)
    return image


def image_preprocess(img_path, norm=True):
    return sim_preprocess(img_path, norm=norm)  
