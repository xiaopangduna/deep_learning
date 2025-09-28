
from ultralytics import YOLO

from ultralytics.utils import RANK, colorstr



from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_cls_dataset, check_det_dataset

from lovely_deep_learning.datasets.yolo_dataset import YoloDataset,read_yolo_detection_labels,read_img


CSV_FILES = [
    "/home/xiaopangdun/project/deep_learning/src/train/datasets/coco8/train.csv"
]  # 可以是相对路径或绝对路径
FIELD_MAP = {
    "img_paths": "data_img",  # 类内字段img对应CSV中的image_path列
    "label_paths": "label_detect_yolo",  # 类内字段label对应CSV中的label_path列
}


# my_dataset = YoloDataset(csv_paths=CSV_FILES, key_map=FIELD_MAP, cache_label_path="cache/coco8_train.cache",cache_image_dir="cache")



path_image = "/home/ubuntu/Desktop/project/deep_learning/src/train/datasets/coco8"

# model = YOLO("pretrained_models/yolov8n.pt")  # 使用下载的配置构建模型
# model.train(data="coco8.yaml", epochs=10,batch=1,workers=0)  # 在COCO数据集上训练模型

cls,bboxes = read_yolo_detection_labels("/home/xiaopangdun/project/deep_learning/src/train/datasets/coco8/labels/train/000000000009.txt")
img,shape_ori = read_img("/home/xiaopangdun/project/deep_learning/src/train/datasets/coco8/images/train/000000000009.jpg","cache/38cb3587f99dab81.npy")


img_path = "/home/xiaopangdun/project/deep_learning/src/train/datasets/coco8/images/train"
imgsz = 640
batch_size = 1
augment = False

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
dataset = YOLODataset(
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
