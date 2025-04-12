import os
import sys

path_parent_dir = os.path.dirname(sys.path[0])
sys.path.insert(0, path_parent_dir)
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torchvision.models import resnet18
import torch_pruning as tp
from src.runner import DeepLearningRunner
from src.datasets import my_datasets
from src.models import my_models
from src.losses import my_losses
from src.metrics import my_metrics





if __name__ == "__main__":
    path_cfgs = r"./configs/home_classify.yml"
    runner = DeepLearningRunner(
        path_cfgs, my_datasets, my_models, my_losses,my_metrics
    )
    runner.train_pruning()
    pass

# device = torch.device("cuda")
# dataset = my_datasets["ClassifyDataset"]
# class_to_index = {"city": 0, "highway": 1}
# path_txt = r"dataset_sample/test.txt"
# cfgs = {}
# cfgs["class_to_index"] = class_to_index
# dataset_train = dataset(path_txt, "train", cfgs)
# dataloader_train = DataLoader(
#     dataset=dataset_train,
#     batch_size=32,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=4,
#     collate_fn=dataset_train.get_collate_fn_for_dataloader(),
# )

# # model = resnet18(pretrained=True)
# # num_ftrs = model.fc.in_features
# # model.fc = nn.Linear(num_ftrs, 2)
# path_model = r"/home/xiaopangdun/project/image_classification_example/logs/training_process/resnet18_important/model_010.pth"
# model = torch.load(path_model)
# print(model)
# device = torch.device("cuda")
# model.to(device)

# example_inputs = torch.randn(1, 3, 224, 224).to(device)
# # 1. Importance criterion
# imp = tp.importance.GroupNormImportance(p=2)  # or GroupTaylorImportance(), GroupHessianImportance(), etc.

# # 2. Initialize a pruner with the model and the importance criterion
# ignored_layers = []
# for m in model.modules():
#     if isinstance(m, torch.nn.Linear) and m.out_features == 2:
#         ignored_layers.append(m)  # DO NOT prune the final classifier!

# pruner = tp.pruner.MetaPruner(  # We can always choose MetaPruner if sparse training is not required.
#     model,
#     example_inputs,
#     importance=imp,
#     pruning_ratio=0.5,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
#     # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
#     ignored_layers=ignored_layers,
# )

# # 3. Prune & finetune the model
# base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
# pruner.step()
# macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
# print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
# # finetune the pruned model here
# # finetune(model)
# # ...
# loss_fn = nn.CrossEntropyLoss()
# # loss_fn = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# for i in range(10):
#     model.train()
#     for net_in, net_out in dataloader_train:
#         datas = list(item["data_tensor"] for item in net_in)
#         datas = torch.stack(datas, 0).to(torch.float32).to(device)
#         labels = list(item["label_tensor"] for item in net_out)
#         labels = torch.stack(labels, 0).to(torch.float32).to(device)
#         outputs = model(datas)
#         loss = loss_fn(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
