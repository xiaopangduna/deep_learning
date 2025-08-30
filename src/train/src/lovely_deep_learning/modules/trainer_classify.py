import os
import json
from pathlib import Path
import argparse
import sys
import logging
import textwrap
import shutil

import lightning as L
import torch
from torch import nn, optim
import torchvision
import torchinfo
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms as T
import torchmetrics
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import torch_pruning as tp
# from pytorch_quantization.tensor_quant import QuantDescriptor
# from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
# from pytorch_quantization import quant_modules
# from pytorch_quantization import nn as quant_nn
# from pytorch_quantization import calib
# import pytorch_quantization
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from src.modules.base_lightning_module import AbstractLightningModule

__all__ = ["ClassifyDataset", "ClassifyModule"]


class ClassifyDataset(Dataset):
    """A basic for building pytorch model input and output tensor

    This class inherits the Dataset(from torch.utils.data) to ensure the way of load dataset ,
    visualize data and label and data enhancemment is same.

    Args:
        path_txt (str): The path of txt file ,whose contents the paths of data and label.
        paths_data (list[str]): A list of paths of data.
        paths_label (list[str]):A list of paths of label.
        transfroms (str): One of train,val,test and  none.Default none
        cfgs (dict): A dictionary holds parameters in data processing.
    """

    def __init__(self, path_txt: str, transfroms: str, cfgs: dict) -> None:
        """Initialize the Dataset.

        Args:
            path_txt (str): The path of txt file ,whose contents the paths of data and label.
            transfroms (str): One of train,val,test and  none.Default none.
            cfgs (dict): A dictionary holds parameters in data processing.
        """
        self.path_txt = path_txt
        self.cfgs = cfgs
        if transfroms == "train":
            self.transforms = self.get_transforms_for_train()
        elif transfroms == "val":
            self.transforms = self.get_transforms_for_val()
        elif transfroms == "test":
            self.transforms = self.get_transforms_for_test()
        else:
            self.transforms = None
        self.paths_data = []
        self.paths_label = []
        with open(self.path_txt, "r") as f:
            for line in f:
                _ = line.split(" ")
                path_data, path_label = _[0], _[1].rstrip()
                self.paths_data.append(path_data)
                self.paths_label.append(path_label)
        assert len(self.paths_label) == len(self.paths_data)
        return

    def __len__(self):
        return len(self.paths_label)

    def __getitem__(self, index):
        net_in, net_out = {}, {}
        path_data = self.paths_data[index]
        path_label = self.paths_label[index]
        data = cv2.imread(path_data)
        if self.transforms:
            transformed = self.transforms(image=data)
            data = transformed["image"]
        with open(path_label, "r") as f:
            labelme = json.load(f)
        flags_raw = labelme["flags"]
        flags_valid = self.convert_raw_to_valid(flags_raw, self.cfgs["class_to_index"])
        flags_tensor = self.convert_valid_to_tensor(flags_valid, self.cfgs["class_to_index"])
        data_tensor = T.ToTensor()(data)
        net_in["img_path"] = path_data
        net_in["img"] = data
        net_in["img_tensor"] = data_tensor
        net_out["target_raw"] = flags_raw
        net_out["target_valid"] = flags_valid
        net_out["target_tensor"] = flags_tensor
        return net_in, net_out

    def get_transforms_for_train(self):
        transforms = A.Compose(
            [
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Rotate(),
                A.OneOf(
                    [
                        A.HueSaturationValue(),
                        A.RGBShift(),
                        A.RandomFog(),
                        A.RandomShadow(),
                        A.RandomRain(),
                    ]
                ),
                A.Resize(224, 224),
            ],
        )
        return transforms

    def get_transforms_for_val(self):
        transforms = A.Compose(
            [
                A.Resize(224, 224),
            ],
        )
        return transforms

    def get_transforms_for_test(self):
        transforms = A.Compose(
            [
                A.Resize(224, 224),
            ],
        )
        return transforms

    def convert_raw_to_valid(self, raw, class_to_index: dict):
        del_keys = set()
        for key in raw.keys():
            if not raw[key] or key not in class_to_index:
                del_keys.add(key)
        for key in del_keys:
            del raw[key]
        return raw

    def convert_valid_to_tensor(self, valid, class_to_index):
        for key in valid.keys():
            flags_tensor = torch.tensor(class_to_index[key]).to(torch.long)
        return flags_tensor

    def draw_tensor_on_image_with_label(self, image: np.ndarray, output, target=None):
        # 绘制预测结果和真值在同一图像上
        class_to_index = self.cfgs["class_to_index"]
        _, index = torch.max(output, dim=0)
        _, target_index = torch.max(target, dim=0)
        for key in class_to_index.keys():
            if class_to_index[key] == index.item():
                class_perdict = key
            if class_to_index[key] == target_index.item():
                class_target = key
        if class_perdict != class_target:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        image = np.ascontiguousarray(image)
        cv2.putText(image, "pred: " + class_perdict, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(image, "true: " + class_target, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        return image

    @staticmethod
    def draw_predict_on_image_without_label(image: np.ndarray, output, cfgs: dict):
        # 绘制预测结果和真值在同一图像上
        class_to_index = cfgs["class_to_index"]
        _, index = torch.max(output, dim=0)
        color = (0, 255, 0)
        for key in class_to_index.keys():
            if class_to_index[key] == index.item():
                class_perdict = key
        image = np.ascontiguousarray(image)
        cv2.putText(image, "pred: " + class_perdict, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        return image

    def get_visual_grid_for_train(self, model):
        visual_indexs = random.sample(range(0, len(self)), self.cfgs["visual_nums"])
        row = col = int(pow(self.cfgs["visual_nums"], 1 / 2))
        images = []
        for i in visual_indexs:
            net_in, net_out = self[i]
            data = net_in["data"]
            data_tensor = net_in["data_tensor"].cuda()
            targer_tesnsor = net_out["label_tensor"]
            model.cuda()
            output = model(data_tensor.unsqueeze(0))
            image = self.draw_tensor_on_image_with_label(data, output.squeeze(), targer_tesnsor)
            image = cv2.resize(image, (300, 300))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        fig, axs = plt.subplots(row, col)
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(images[i + j])
                axs[i, j].axis("off")
        # plt.tight_layout()
        # plt.savefig("bb.jpg")
        # for i in range(9):
        #     image = images[i]
        #     cv2.imwrite(r"{}.jpg".format(i),image)
        return fig

    @staticmethod
    def get_collate_fn_for_dataloader():
        def collate_fn(x):
            return list(zip(*x))

        return collate_fn


    def generate_titles(image_paths, titles=None):
        """
        根据 image_paths 和 titles 生成最终的 titles 列表

        参数:
            image_paths (list): 包含图片路径（字符串）或图像数据（np.ndarray）的列表。
            titles (list): 包含要绘制在图像左上角的字符串的列表，默认为 None。

        返回:
            list: 生成的 titles 列表。
        """
        num_images = len(image_paths)
        if titles is None:
            titles = []  # 初始化 titles 为空列表
            for idx, image_data in enumerate(image_paths):
                if isinstance(image_data, str):
                    # 如果是图片路径，提取文件名
                    titles.append(os.path.basename(image_data))
                elif isinstance(image_data, np.ndarray):
                    # 如果是 np.ndarray，使用序号
                    titles.append(f"Image {idx + 1}")
                else:
                    raise TypeError("image_paths 中的元素必须是字符串（路径）或 np.ndarray（图像数据）")
        else:
            # 如果 titles 的长度小于图片数量，按顺序补充缺失的部分
            if len(titles) < num_images:
                for idx in range(len(titles), num_images):
                    if isinstance(image_paths[idx], str):
                        # 如果是图片路径，提取文件名
                        titles.append(os.path.basename(image_paths[idx]))
                    elif isinstance(image_paths[idx], np.ndarray):
                        # 如果是 np.ndarray，使用序号
                        titles.append(f"Image {idx + 1}")
                    else:
                        raise TypeError("image_paths 中的元素必须是字符串（路径）或 np.ndarray（图像数据）")
        return titles

    def calculate_subplot_layout(num_images, nrows=None, ncols=None):
        """
        计算子图的行数和列数

        参数:
            num_images (int): 图片的数量。
            nrows (int): 子图的行数，默认为 None。
            ncols (int): 子图的列数，默认为 None。

        返回:
            tuple: (nrows, ncols) 子图的行数和列数。
        """
        if nrows is None and ncols is None:
            # 自动计算 nrows 和 ncols
            ncols = math.ceil(math.sqrt(num_images))  # 列数为平方根向上取整
            nrows = math.ceil(num_images / ncols)     # 行数为总数除以列数向上取整
        elif nrows is None:
            # 仅指定 ncols，自动计算 nrows
            nrows = math.ceil(num_images / ncols)
        elif ncols is None:
            # 仅指定 nrows，自动计算 ncols
            ncols = math.ceil(num_images / nrows)
        return nrows, ncols

    def plot_images(axes, image_paths, titles, nrows, ncols):
        """
        在子图中绘制图片并添加标题

        参数:
            axes (np.ndarray): 子图的坐标轴数组。
            image_paths (list): 包含图片路径（字符串）或图像数据（np.ndarray）的列表。
            titles (list): 包含要绘制在图像左上角的字符串的列表。
            nrows (int): 子图的行数。
            ncols (int): 子图的列数。
        """
        num_images = len(image_paths)
        for idx in range(nrows * ncols):
            row = idx // ncols  # 计算当前子图的行索引
            col = idx % ncols   # 计算当前子图的列索引
            ax = axes[row, col]

            if idx < num_images:
                # 获取当前图像数据或路径
                image_data = image_paths[idx]

                # 判断输入类型
                if isinstance(image_data, str):
                    # 如果是字符串，读取图片
                    image = cv2.imread(image_data)
                    if image is None:
                        raise ValueError(f"无法读取图片: {image_data}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB
                elif isinstance(image_data, np.ndarray):
                    # 如果是 np.ndarray，直接使用
                    image = image_data
                else:
                    raise TypeError("image_paths 中的元素必须是字符串（路径）或 np.ndarray（图像数据）")

                # 显示图片
                ax.imshow(image)

                # 在图像左上角绘制 titles 中的字符串
                ax.text(
                    10,  # x 坐标（距离左侧 10 像素）
                    20,  # y 坐标（距离顶部 20 像素）
                    titles[idx],  # 要绘制的文本
                    color='white',  # 文本颜色
                    fontsize=12,    # 字体大小
                    backgroundcolor='black'  # 背景颜色
                )
            else:
                # 显示空白图
                ax.axis('off')  # 关闭坐标轴

            ax.axis('off')  # 关闭坐标轴

    def save_images(image_paths, titles, dir_save):
        """
        保存图片到指定文件夹

        参数:
            image_paths (list): 包含图片路径（字符串）或图像数据（np.ndarray）的列表。
            titles (list): 包含要绘制在图像左上角的字符串的列表。
            dir_save (str): 保存图片的文件夹路径。
        """
        for idx, image_data in enumerate(image_paths):
            # 获取当前图像数据或路径
            if isinstance(image_data, str):
                # 如果是字符串，读取图片
                image = cv2.imread(image_data)
                if image is None:
                    raise ValueError(f"无法读取图片: {image_data}")
            elif isinstance(image_data, np.ndarray):
                # 如果是 np.ndarray，直接使用
                image = image_data
            else:
                raise TypeError("image_paths 中的元素必须是字符串（路径）或 np.ndarray（图像数据）")

            # 在图像上绘制 titles 中的字符串
            image_with_text = image.copy()
            cv2.putText(
                image_with_text,
                titles[idx],
                (10, 30),  # 文本位置（距离左侧 10 像素，距离顶部 30 像素）
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
                1,  # 字体大小
                (255, 255, 255),  # 文本颜色（白色）
                2,  # 文本厚度
                cv2.LINE_AA  # 抗锯齿
            )

            # 保存图片
            save_path = os.path.join(dir_save, f"image_{idx + 1}.png")
            cv2.imwrite(save_path, cv2.cvtColor(image_with_text, cv2.COLOR_RGB2BGR))
            print(f"图片已保存至: {save_path}")

    def plot_image_subplots_flexible(image_paths, titles=None, nrows=None, ncols=None, figsize=(10, 8), dir_save=None):
        """
        灵活绘制子图布局并显示图片，将 titles 中的字符串绘制在图像左上角，并支持保存图片

        参数:
            image_paths (list): 包含图片路径（字符串）或图像数据（np.ndarray）的列表。
            titles (list): 包含要绘制在图像左上角的字符串的列表，默认为 None。
            nrows (int): 子图的行数，默认为 None。
            ncols (int): 子图的列数，默认为 None。
            figsize (tuple): 整个画布的大小，默认为 (10, 8)。
            dir_save (str): 保存图片的文件夹路径，默认为 None（不保存）。
        """
        # 检查输入数据的长度
        num_images = len(image_paths)

        # 生成 titles
        titles = generate_titles(image_paths, titles)

        # 如果指定了保存路径，检查文件夹是否存在，如果不存在则创建
        if dir_save is not None:
            os.makedirs(dir_save, exist_ok=True)

        # 计算子图的行数和列数
        nrows, ncols = calculate_subplot_layout(num_images, nrows, ncols)

        # 创建子图
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # 如果只有一个子图，将 axes 转换为二维数组以便统一处理
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)

        # 绘制图片
        plot_images(axes, image_paths, titles, nrows, ncols)

        # 调整子图之间的间距
        plt.tight_layout()

        # 显示图形
        plt.show()

        # 如果指定了保存路径，保存图片
        if dir_save is not None:
            save_images(image_paths, titles, dir_save)

# # 示例数据
# # 1. 图片路径
# image_paths = [
#     'path/to/image1.jpg',
#     'path/to/image2.jpg',
#     'path/to/image3.jpg',
#     'path/to/image4.jpg'
# ]

# # 2. 图像数据（np.ndarray）
# image_data_1 = np.random.rand(100, 100, 3)  # 随机生成一张 100x100 的 RGB 图像
# image_data_2 = np.random.rand(100, 100, 3)  # 随机生成一张 100x100 的 RGB 图像

# # 混合输入
# mixed_image_paths = [
#     'path/to/image1.jpg',  # 图片路径
#     image_data_1,          # 图像数据
#     'path/to/image2.jpg',  # 图片路径
#     image_data_2           # 图像数据
# ]

# # 调用函数绘制图形
# # 1. 自动布局，不保存图片
# plot_image_subplots_flexible(mixed_image_paths)

# # 2. 指定 nrows=2, ncols=2，保存图片
# plot_image_subplots_flexible(mixed_image_paths, dir_save="output_images")

# L.LightningModule
class ClassifyModule(AbstractLightningModule):
    def __init__(
        self,
        model_name: str,
        model_hparams: dict,
        optimizer_name: str,
        optimizer_hparams: dict,
    ):
        """CIFARModule.

        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = self.set_model()
        self.pruner = None
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.input_size = self.hparams.model_hparams["input_size"]
        self.example_input_array = torch.zeros(self.input_size, dtype=torch.float32).to(self.device)

        self.metrics = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.hparams.model_hparams["num_classes"], average="micro"
        )

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        net_in, net_out = batch
        imgs = list(item["img_tensor"] for item in net_in)
        imgs = torch.stack(imgs, 0)
        targets = list(item["target_tensor"] for item in net_out)
        targets = torch.stack(targets, 0)
        preds = self.model(imgs)
        loss = self.loss_module(preds, targets)
        acc = self.metrics(preds, targets)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = list(item["img_tensor"] for item in net_in)
        imgs = torch.stack(imgs, 0)
        targets = list(item["target_tensor"] for item in net_out)
        targets = torch.stack(targets, 0)
        preds = self.model(imgs)
        acc = self.metrics(preds, targets)
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = list(item["img_tensor"] for item in net_in)
        imgs = torch.stack(imgs, 0)
        targets = list(item["target_tensor"] for item in net_out)
        targets = torch.stack(targets, 0)
        preds = self.model(imgs)
        acc = self.metrics(preds, targets)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0, *args, **kwargs):
        # TODO
        net_in, net_out = batch
        imgs = list(item["img_tensor"] for item in net_in)
        imgs = torch.stack(imgs, 0)
        preds = self.model(imgs)
        if net_out:
            targets = list(item["target_tensor"] for item in net_out)
            targets = torch.stack(targets, 0)
        else:
            targets = None
        return preds, targets,

    def set_model(self):
        if self.hparams.model_name == "mobilenet_v3_small":
            model = torchvision.models.mobilenet_v3_small()
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_ftrs, self.hparams.model_hparams["num_classes"])
        elif self.hparams.model_name == "resnet18":
            model = torchvision.models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.hparams.model_hparams["num_class"])
        else:
            pass
        return model

    def set_pruner(self, pruning_ratio=0.5, iterative_steps=1):
        # 1. Importance criterion
        imp = tp.importance.GroupNormImportance(p=2)  # or GroupTaylorImportance(), GroupHessianImportance(), etc.
        # 2. Initialize a pruner with the model and the importance criterion
        ignored_layers = []
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == self.hparams.model_hparams["num_classes"]:
                ignored_layers.append(m)  # DO NOT prune the final classifier!
        self.pruner = tp.pruner.MetaPruner(  # We can always choose MetaPruner if sparse training is not required.
            self.model.to(self.device),
            self.example_input_array.to(self.device),
            iterative_steps=iterative_steps,
            importance=imp,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
        )
        return

    def set_quantization_init(self):
        quant_modules.initialize()
        quant_desc_input = QuantDescriptor(calib_method="histogram")
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        return

    def set_quantization_ptq(self, dataset):
        def collect_stats(model, data_loader, num_batches):
            """Feed data to the network and collect statistic"""

            # Enable calibrators
            for name, module in model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.disable_quant()
                        module.enable_calib()
                    else:
                        module.disable()
            # net_in, net_out = batch
            # imgs = list(item["img_tensor"] for item in net_in)
            # imgs = torch.stack(imgs, 0)
            # targets = list(item["target_tensor"] for item in net_out)
            # targets = torch.stack(targets, 0)
            for i, (net_in, net_out) in tqdm(enumerate(data_loader), total=num_batches):
                imgs = list(item["img_tensor"] for item in net_in)
                image = torch.stack(imgs, 0)
                model(image.cuda())
                if i >= num_batches:
                    break

            # Disable calibrators
            for name, module in model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.enable_quant()
                        module.disable_calib()
                    else:
                        module.enable()

        def compute_amax(model, **kwargs):
            # Load calib result
            for name, module in model.named_modules():
                if isinstance(module, quant_nn.TensorQuantizer):
                    if module._calibrator is not None:
                        if isinstance(module._calibrator, calib.MaxCalibrator):
                            module.load_calib_amax()
                        else:
                            module.load_calib_amax(**kwargs)
                    print(f"{name:40}: {module}")
            model.cuda()

        data_loader = DataLoader(dataset=dataset, collate_fn=dataset.get_collate_fn_for_dataloader(), batch_size=512)
        with torch.no_grad():
            collect_stats(self.model, data_loader, num_batches=2)
            compute_amax(self.model, method="percentile", percentile=99.99)
        return

    def on_save_checkpoint(self, checkpoint):
        if self.pruner is None:
            checkpoint["full_state_dict"] = None
            checkpoint["attributions"] = None
        else:
            state_dict = tp.state_dict(self.model)
            checkpoint["full_state_dict"] = state_dict["full_state_dict"]
            checkpoint["attributions"] = state_dict["attributions"]
        path_save = Path(self.logger.log_dir + "/checkpoints/last.pth")
        path_save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, str(path_save))

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("full_state_dict") and checkpoint.get("attributions"):
            self.model = tp.load_state_dict(self.model, state_dict=checkpoint)
        return super().on_load_checkpoint(checkpoint)

    @staticmethod
    def export(model, path_save: str, mode: str, input_size: list, is_quantization: bool = False):
        if mode == "onnx":
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(input_size, device=device)
            input_names = ["input"]
            output_names = ["output"]
            if is_quantization:
                path_save = path_save.replace(".onnx", "_quant.onnx")
                quant_modules.initialize()
                quant_desc_input = QuantDescriptor(calib_method="histogram")
                quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
                quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
                quant_nn.TensorQuantizer.use_fb_fake_quant = True
                with torch.no_grad():
                    # enable_onnx_checker needs to be disabled. See notes below.
                    torch.onnx.export(
                        model,
                        dummy_input,
                        path_save,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=False,
                        opset_version=13,
                    )
                    quant_nn.TensorQuantizer.use_fb_fake_quant = False

            else:
                torch.onnx.export(
                    model,
                    dummy_input,
                    path_save,
                    input_names=input_names,
                    output_names=output_names,
                    verbose=False,
                    opset_version=13,
                )

        return


if __name__ == "__main__":
    pass
