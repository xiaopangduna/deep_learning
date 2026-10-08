"""CSV datasets. Each sample is ``(net_in, net_out)``."""

from .base import BaseDataset
from .box_crop_classifier import BoxCropClassifierDataset
from .image_classifier import ImageClassifierDataset
from .object_detect import ObjectDetectDataset

__all__ = [
    "BaseDataset",
    "BoxCropClassifierDataset",
    "ImageClassifierDataset",
    "ObjectDetectDataset",
]
