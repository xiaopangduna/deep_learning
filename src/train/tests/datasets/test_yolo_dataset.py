from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils import RANK, colorstr

from typing import Any, Dict, List, Optional, Tuple

from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset as YOLODatasetU
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data.augment import (
    Compose,
    Format,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)

from lovely_deep_learning.datasets.yolo_dataset import YoloDataset, read_yolo_detection_labels, read_img


CSV_FILES = ["datasets/coco8/train.csv"]  # 可以是相对路径或绝对路径
FIELD_MAP = {
    "img_paths": "data_img",  # 类内字段img对应CSV中的image_path列
    "label_paths": "label_detect_yolo",  # 类内字段label对应CSV中的label_path列
}

hpy= DEFAULT_CFG


# def build_transforms(augment, hyp: Optional[Dict] = None) -> Compose:
#     """
#     Build and append transforms to the list.

#     Args:
#         hyp (dict, optional): Hyperparameters for transforms.

#     Returns:
#         (Compose): Composed transforms.
#     """
#     if augment:
#         hyp.mosaic = hyp.mosaic if augment and not self.rect else 0.0
#         hyp.mixup = hyp.mixup if augment and not self.rect else 0.0
#         hyp.cutmix = hyp.cutmix if augment and not self.rect else 0.0
#         transforms = v8_transforms(self, self.imgsz, hyp)
#     else:
#         transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
#     transforms.append(
#         Format(
#             bbox_format="xywh",
#             normalize=True,
#             return_mask=self.use_segments,
#             return_keypoint=self.use_keypoints,
#             return_obb=self.use_obb,
#             batch_idx=True,
#             mask_ratio=hyp.mask_ratio,
#             mask_overlap=hyp.overlap_mask,
#             bgr=hyp.bgr if augment else 0.0,  # only affect training.
#         )
#     )
#     return transforms




# model = YOLO("pretrained_models/yolov8n.pt")  # 使用下载的配置构建模型
# model.train(data="coco8.yaml", epochs=10,batch=1,workers=0)  # 在COCO数据集上训练模型

img_path = "datasets/coco8/images/train"
imgsz = 640
batch_size = 1
augment = False # *"train"

rect = False
cache = None
single_cls = False
stride = 32
pad = 0.0
prefix = colorstr("train: ")
task = "detect"
classes = None
fraction = 1.0

cfg = DEFAULT_CFG
overrides = {
    "task": "detect",
    "data": "coco8.yaml",
    "imgsz": 640,
    "single_cls": False,
    "model": "pretrained_models/yolov8n.pt",
    "epochs": 10,
    "batch": 1,
    "workers": 0,
    "mode": "train",
}
args = get_cfg(cfg, overrides)
hyp = cfg
data = check_det_dataset("coco8.yaml")
dataset = YOLODatasetU(
    img_path=img_path,
    imgsz=imgsz,
    batch_size=batch_size,
    augment=augment,
    hyp=hyp,
    rect=rect,
    cache=cache,
    single_cls=single_cls,
    stride=stride,
    pad=pad,
    prefix=prefix,
    task=task,
    classes=classes,
    data=data,
    fraction=fraction,
)
dataset[0]
pass
transforms = Compose([LetterBox(new_shape=(640, 640), scaleup=False)])
transforms.append(
    Format(
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        batch_idx=True,
        mask_ratio=hyp.mask_ratio,
        mask_overlap=hyp.overlap_mask,
        bgr= 0.0,  # only affect training.
    ))
my_dataset = YoloDataset(
    csv_paths=CSV_FILES, key_map=FIELD_MAP,transform=transforms, cache_label_path="cache/coco8_train.cache", cache_image_dir="cache"
)

sample= my_dataset[0]
print(sample)
img = cv2.imread(sample["img_path"])
img_npy = np.load(sample["img_npy_path"])
img_with_box = YoloDataset.draw_bounding_boxes(img, sample["bboxes"], sample["classes"])
cv2.imwrite("/home/xiaopangdun/project/deep_learning/src/train/tmp/test.jpg", img_with_box)

pass