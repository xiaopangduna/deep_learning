"""CSV datasets. Each sample is ``(net_in, net_out)``."""

from .attribute_classifier import AttributeFromDetectBoxDataset
from .base import BaseDataset
from .image_classifier import ImageClassifierDataset, ImageClassifierFromDetectBoxDataset
from .object_detect import ObjectDetectDataset

__all__ = [
    "AttributeFromDetectBoxDataset",
    "BaseDataset",
    "ImageClassifierDataset",
    "ImageClassifierFromDetectBoxDataset",
    "ObjectDetectDataset",
]
