from ultralytics import YOLO

model = YOLO("/home/xiaopangdun/project/deep_learning/src/train/configs/object_detection/yolov8.yaml")  # 使用下载的配置构建模型
model.train(data="coco8.yaml", epochs=10)