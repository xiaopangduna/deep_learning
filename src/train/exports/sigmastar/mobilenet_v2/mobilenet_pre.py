# -*- coding: utf-8 -*-
import os

from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision.transforms import transforms


def tv_standard_preprocess(image_file, input_size=224, norm=True):
    ori_image = Image.open(image_file).convert('RGB')
    if input_size == 299:
        scaled_size = input_size
    else:
        scaled_size = 256
    image = transforms.Resize(scaled_size)(ori_image)
    image = transforms.CenterCrop(input_size)(image)
    if norm:
        image = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ]
        )(image)
    else:
        image = np.array(image)
        # image = np.expand_dims(image, 0).astype(np.uint8)
        image = torch.from_numpy(image)
        image = image.permute((2, 0, 1))
    image = torch.unsqueeze(image, 0)
    return image


def sim_standard_preprocess(image_file, input_size=224, norm=True):
    ori_image = Image.open(image_file).convert('RGB')
    if input_size == 299:
        scaled_size = input_size
    else:
        scaled_size = 256
    image = transforms.Resize(scaled_size)(ori_image)
    image = transforms.CenterCrop(input_size)(image)
    if norm:
        image = transforms.Compose(
            [
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ]
        )(image)
        image = np.array(image)
        image = np.expand_dims(image, 0).astype(np.float32)
        image = np.transpose(image, axes=(0, 2, 3, 1)).copy()
    else:
        image = np.array(image)
        image = np.expand_dims(image, 0).astype(np.uint8)
    return image


def tv_classification_visualize(image_file, processed_output, k=5):
    if isinstance(processed_output, np.ndarray):
        processed_output = torch.from_numpy(processed_output)
    predictions = nn.functional.softmax(processed_output, dim=1)
    predictions = predictions.detach().cpu().numpy().flatten()
    topk = np.argsort(predictions)[-1:-(k+1):-1]
    print('Image File: {}'.format(os.path.basename(image_file)))
    print('Top{} Predictions:'.format(k))
    for i in range(len(topk)):
        print('{:d} {:.2%}'.format(topk[i], predictions[topk[i]]))


def onnx_preprocess(image_file, input_size=224):
    torch_processed = torch_preprocess(image_file, input_size)
    onnx_processed = np.array(torch_processed)
    return onnx_processed


def onnx_postprocess(model_outputs, **kwargs):
    processed_outputs = model_outputs
    if 'visualize' in kwargs and kwargs['visualize'] is True:
        assert 'image_file' in kwargs
        torch_processed = torch.from_numpy(processed_outputs)
        tv_classification_visualize(kwargs['image_file'], torch_processed, k=5)
    return processed_outputs

def torch_preprocess(image_file, input_size=224):
    processed_image = tv_standard_preprocess(image_file, input_size)
    return processed_image


def torch_postprocess(model_outputs, **kwargs):
    processed_outputs = model_outputs
    if 'visualize' in kwargs and kwargs['visualize'] is True:
        assert 'image_file' in kwargs
        tv_classification_visualize(kwargs['image_file'], processed_outputs, k=5)
    return processed_outputs


def sim_preprocess(image_file, input_size=224, norm=True):
    sim_processed = sim_standard_preprocess(image_file, input_size, norm=norm)
    return sim_processed


def sim_postprocess(model_outputs, **kwargs):
    processed_outputs = model_outputs
    if 'visualize' in kwargs and kwargs['visualize'] is True:
        assert 'image_file' in kwargs
        tv_classification_visualize(kwargs['image_file'], processed_outputs, k=5)
    return processed_outputs


def image_preprocess(img_path, norm=True):
    return sim_preprocess(img_path, norm=norm)
