# -*- encoding: utf-8 -*-
"""
@File    :   object_detection.py
@Python  :   python3.8
@version :   0.0
@Time    :   2025/03/03 21:31:14
@Author  :   xiaopangdun
@Email   :   18675381281@163.com
@Desc    :   This is a simple example
"""
import os
import json
from pathlib import Path
import random
import argparse
import sys
import logging
import textwrap
import shutil

import lightning as L
import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
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
import torch.profiler
from src.models.object_detection import DirectionalCornerDetectionModel

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
from src.losses.object_detection import DirectionalCornerDetectionLoss
from src.metrics.object_detection import DirectionalCornerDetectionMetric
from src.models.dmpr import DirectionalPointDetector
from src.datasets.object_detection import DirectionalCornerDetectionDataset
from src.utils.profiler_pytorch import LayerTimeProfiler


class DirectionalCornerDetectionModule(AbstractLightningModule):
    def __init__(
        self,
        model_name: str,
        model_hparams: dict,
        optimizer_name: str,
        optimizer_hparams: dict,
        loss_name: str = "DirectionalCornerDetectionLoss",
        loss_hparams: dict = {"num_classes": 2},
        metrics_name: str = "DirectionalCornerDetectionMetric",
        metrics_hparams: dict = {
            "class_to_index": {"T": 0, "L": 1},
            "confidence_threshold": 0.5,
            "distance_threshold": 0.1,
            "angle_threshold": 5,
            "consider_class": False,
        },
        dir_save_predict_images: str = None,
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
        # 随机保存图像的batch索引
        self.batch_idx_save_image_with_keypoint = 0
        # Create model
        self.model = self.set_model()
        self.pruner = None
        # Create loss module
        self.loss_module = DirectionalCornerDetectionLoss(**loss_hparams)
        # Example input for visualizing the graph in Tensorboard
        self.input_size = self.hparams.model_hparams["input_size"]
        self.example_input_array = torch.randn(self.input_size, dtype=torch.float32).to(self.device)
        self.metrics = DirectionalCornerDetectionMetric(metrics_hparams)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.model.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.model.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # print("可训练参数列表:")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: shape={tuple(param.shape)}")
        # print("优化器参数数量:", len(list(optimizer.param_groups[0]['params'])))
        # total_optimizer_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        # print("优化器管理的参数总数:", total_optimizer_params)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        net_in, net_out = batch
        imgs = list(item["img_tensor"] for item in net_in)
        imgs = torch.stack(imgs, 0)
        targets = list(item["keypoint_tensor"] for item in net_out)
        targets = torch.stack(targets, 0)
        preds = self.model(imgs)
        loss = self.loss_module(preds, targets)
        metrics = self.metrics(preds, targets)
        prefixed_metrics = {f"{key}/train": value for key, value in metrics.items()}
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        # self.log("train_accuracy", metrics["accuracy"], on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(prefixed_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        net_in, net_out = batch
        imgs = list(item["img_tensor"] for item in net_in)
        imgs = torch.stack(imgs, 0)
        targets = list(item["keypoint_tensor"] for item in net_out)
        targets = torch.stack(targets, 0)
        preds = self.model(imgs)
        loss = self.loss_module(preds, targets)
        metrics = self.metrics(preds, targets)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        prefixed_metrics = {f"{key}/val": value for key, value in metrics.items()}
        self.log_dict(prefixed_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        try:
            if batch_idx == self.batch_idx_save_image_with_keypoint:
                img_with_keypoint = DirectionalCornerDetectionDataset.draw_tensor_on_data(imgs[0], targets[0], preds[0])
                # image_tensor = torch.from_numpy(img_with_keypoint).permute(2, 0, 1).float() / 255.0
                image_tensor = DirectionalCornerDetectionDataset.convert_image_from_numpy_to_tensor(img_with_keypoint)
                self.logger.experiment.add_image(f"val/prediction", image_tensor, self.current_epoch)
        except Exception as e:
            print("failed to draw image")

    def on_train_epoch_end(self):
        self.batch_idx_save_image_with_keypoint = random.randint(0, self.trainer.fit_loop.max_batches - 1)
        return

    def on_validation_epoch_end(self):
        # 修改batch_idx_save_image_with_keypoint的值，实现随机抽图功能
        self.batch_idx_save_image_with_keypoint = random.randint(0, self.trainer.fit_loop.max_batches - 1)
        return

    def test_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        net_in, net_out = batch
        imgs = list(item["img_tensor"] for item in net_in)
        imgs = torch.stack(imgs, 0)
        targets = list(item["keypoint_tensor"] for item in net_out)
        targets = torch.stack(targets, 0)
        preds = self.model(imgs)
        loss = self.loss_module(preds, targets)
        metrics = self.metrics(preds, targets)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        # self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        # self.log("test_accuracy", metrics["accuracy"], on_epoch=True)
        # self.log("test_precision", metrics["precision"], on_epoch=True)
        # self.log("test_recall", metrics["recall"], on_epoch=True)
        # self.log("test_f1_score", metrics["f1_score"], prog_bar=True)
        # self.log("test_confidence_mean", metrics["confidence_mean"], on_epoch=True)
        # self.log("test_angle_error_mean", metrics["angle_error_mean"], on_epoch=True)
        # self.log("test_angle_error_max", metrics["angle_error_max"], on_epoch=True)
        # self.log("test_center_error_mean", metrics["center_error_mean"], prog_bar=True)
        # self.log("test_center_error_max", metrics["center_error_max"], on_epoch=True)
        # 如果需要，可以记录类别分布
        # for class_name, values in metrics["class_distribution"].items():
        #     self.log(f"val_class_{class_name}_predicted", values["predicted"], on_epoch=True)
        #     self.log(f"val_class_{class_name}_target", values["target"], on_epoch=True)

    def predict_step(self, batch, batch_idx):
        # TODO
        net_in, net_out = batch
        imgs = list(item["img_tensor"] for item in net_in)
        imgs = torch.stack(imgs, 0)
        preds = self.model(imgs)
        targets = list(item["keypoint_tensor"] for item in net_out)
        targets = torch.stack(targets, 0)
        batch_size, _, H, W = imgs.shape
        for i in range(batch_size):
            path_img = Path(net_in[i]["img_path"])
            img_with_keypoint = DirectionalCornerDetectionDataset.draw_tensor_on_data(imgs[i], targets[i], preds[i])
            path_save_img = f"{self.hparams.dir_save_predict_images}/{path_img.stem}.jpg"
            if not os.path.exists(path_save_img):
                cv2.imwrite(path_save_img, img_with_keypoint)
            else:
                print(f"file {path_save_img} already exists")

        print(f"save image to {self.hparams.dir_save_predict_images}")
        return preds, targets

    def on_validation_epoch_end(self):
        # 获取图片
        # 绘制图片
        # 保存图片

        return

    def get_model(self):
        if self.hparams.model_name == "DMPR":
            model = DirectionalPointDetector(
                3, self.hparams.model_hparams["depth_factor"], self.hparams.model_hparams["feature_map_channel"]
            )
        else:
            pass
        return model

    def set_model(self):
        if self.hparams.model_name == "DMPR":
            model = DirectionalPointDetector(
                3, self.hparams.model_hparams["depth_factor"], self.hparams.model_hparams["feature_map_channel"]
            )
        elif self.hparams.model_name == "DirectionalCornerDetectionModel":
            model = DirectionalCornerDetectionModel(**self.hparams.model_hparams)

            pass
        return model

    def get_pruner(self, pruning_ratio=0.5, iterative_steps=1):
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
    # def profile(self,use_gpu=True,warmup_runs=3,actual_runs=10,save_folder=None):
    #     if save_folder is None:
    #         save_folder = self.logger.log_dir
    #     profiler = LayerTimeProfiler(self.model, use_gpu=use_gpu, warmup_runs=warmup_runs, actual_runs=actual_runs,
    #                                      save_folder=save_folder)
    #     profiler.profile(self.example_input_array)
    #     profiler.generate_report(self.hparams.model_hparams["backbone_name"])


        # return 