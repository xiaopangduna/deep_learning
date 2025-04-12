import os
import sys

path_parent_dir = os.path.dirname(sys.path[0])
sys.path.insert(0, path_parent_dir)
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.runner import DeepLearningRunner
from src.datasets import my_datasets
from src.models import my_models
from src.losses import my_losses
from src.metrics import my_metrics

path_cfgs = r"D:\A_Project\image_classification_example\configs\home_classify.yml"
path_model = (
    r"D:\A_Project\image_classification_example\logs\training_process\resnet18_2024-07-21_20-15-10\model_080.pth"
)
runner = DeepLearningRunner(path_cfgs, my_datasets, my_models, my_losses, my_metrics)
dataset = runner.datasets["test"]
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=dataset.get_collate_fn_for_dataloader(),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path_model)
transform_fn = dataset.get_transforms_for_test()
dir_save = Path(r"")
dir_images = Path(r"")
path_images = [path for path in dir_images.iterdir()]
_, report = runner.inference(model, transform_fn, path_images, dir_save, device)
print(report)
