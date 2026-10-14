"""CSV datasets. Each sample is ``(net_in, net_out)``."""

from .base import BaseDataset
from .image_classifier import ImageClassifierDataset, ImageClassifierFromDetectBoxDataset
from .object_detect import ObjectDetectDataset

__all__ = [
    "BaseDataset",
    "ImageClassifierDataset",
    "ImageClassifierFromDetectBoxDataset",
    "ObjectDetectDataset",
]
