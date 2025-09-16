from ultralytics import YOLO


model = YOLO("pretrained_models/yolov8n.pt")  # 使用下载的配置构建模型
model.train(data="coco8.yaml", epochs=10)