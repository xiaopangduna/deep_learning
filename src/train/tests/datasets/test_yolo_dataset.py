from ultralytics import YOLO
# from ultralytics.models import yolo
# from ultralytics.utils import (
#     ARGV,
#     ASSETS,
#     DEFAULT_CFG_DICT,
#     LOGGER,
#     RANK,
#     SETTINGS,
#     YAML,
#     callbacks,
#     checks,
# )

model = YOLO("pretrained_models/yolov8n.pt")  # 使用下载的配置构建模型
model.train(data="coco8.yaml", epochs=10,batch=1,workers=0)  # 在COCO数据集上训练模型


# dataset_path = "/home/xiaopangdun/project/yolo/datasets/coco8/images/train"
# mode = "train"
# rank = -1
# batch_size = 1
# args = {
#     "task": "detect",
#     "data": "coco8.yaml",
#     "imgsz": 640,
#     "single_cls": False,
#     "model": "pretrained_models/yolov8n.pt",
#     "epochs": 10,
#     "batch": 1,
#     "mode": "train",
# }

# # self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
# # self.model = self.trainer.model
# my_callbacks = callbacks.get_default_callbacks()
# trainer =yolo.detect.DetectionTrainer(overrides=args, _callbacks=my_callbacks)
# trainer.model = trainer.get_model(weights=model.model if self.ckpt else None, cfg=self.model.yaml)
# dataloader = trainer.get_dataloader(dataset_path=dataset_path, mode=mode, rank=rank, batch_size=batch_size)
# for batch in dataloader:
#     pass
# pass