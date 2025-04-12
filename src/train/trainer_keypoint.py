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

from src.datasets.keypoint import RegressionPointDataset
from src.models.dmpr import DirectionalPointDetector
from src.metrics.keypoint import EuclideanDistance


class KeypointModule(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
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
        self.model = self.get_model()
        # self.dataset = 
        # self.logger =
        # Create loss module
        self.loss_module = nn.MSELoss()
        # Example input for visualizing the graph in Tensorboard
        # TODO
        self.input_size = self.hparams.model_hparams["input_size"]
        self.example_input_array = torch.zeros(self.input_size, dtype=torch.float32).to(self.device)
        # self.metrics = torchmetrics.Accuracy(
        #     task="multiclass", num_classes=self.hparams.model_hparams["num_classes"], average="micro"
        # )
        self.metrics = EuclideanDistance()

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
        targets = list(item["keypoint_tensor"] for item in net_out)
        targets = torch.stack(targets, 0)
        preds = self.model(imgs)
        loss = self.loss_module(preds, targets)
        # 把输出改成valid格式再计算位置误差等参数
        # TODO 将self.metrics拼接
        for i in range(preds.shape[0]):
            tmp_pred = RegressionPointDataset.convert_tensor_to_valid(preds[i])
            tmp_target = RegressionPointDataset.convert_tensor_to_valid(targets[i])
            metrics = self.metrics(tmp_pred,  tmp_target)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_distance_error", metrics["distance_error"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss)
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
        # 把输出改成valid格式再计算位置误差等参数
        # TODO 将self.metrics拼接
        for i in range(preds.shape[0]):
            tmp_pred = RegressionPointDataset.convert_tensor_to_valid(preds[i])
            tmp_target = RegressionPointDataset.convert_tensor_to_valid(targets[i])
            metrics = self.metrics(tmp_pred,  tmp_target)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("val_distance_error", metrics["distance_error"], on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        net_in, net_out = batch
        imgs = list(item["img_tensor"] for item in net_in)
        imgs = torch.stack(imgs, 0)
        targets = list(item["keypoint_tensor"] for item in net_out)
        targets = torch.stack(targets, 0)
        preds = self.model(imgs)
        acc = self.metrics(preds, targets)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def get_model(self):
        if self.hparams.model_name == "DMPR":
            model = DirectionalPointDetector(
                3, self.hparams.model_hparams["depth_factor"], self.hparams.model_hparams["feature_map_channel"]
            )
        else:
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

        pruner = tp.pruner.MetaPruner(  # We can always choose MetaPruner if sparse training is not required.
            self.model.to(self.device),
            self.example_input_array.to(self.device),
            iterative_steps=iterative_steps,
            importance=imp,
            pruning_ratio=pruning_ratio,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            ignored_layers=ignored_layers,
        )

        return pruner

    def get_quantization_model():
        pass


def cmd_train(args):
    module = KeypointModule(**args.module_cfgs)
    if args.mode == "train":
        name = module.hparams.model_name
        version = None
        args.ckpt_path, args.path_model = None, None
        args.pruning_cfgs = None
    elif args.mode == "train_resume":
        if args.path_model:
            model = torch.load(args.path_model)
            module.model = model
        args.default_roor_dir = Path(args.ckpt_path).parts[-4]
        name = Path(args.ckpt_path).parts[-3]
        version = Path(args.ckpt_path).parts[-2]
    elif args.mode == "train_pruning":
        if args.path_model:
            model = torch.load(args.path_model)
            module.model = model
        args.default_roor_dir = Path(args.ckpt_path).parts[-4]
        name = Path(args.ckpt_path).parts[-3] + "_pruning"
        version = Path(args.ckpt_path).parts[-2]
    elif args.mode == "train_quantization":
        if args.path_model:
            model = torch.load(args.path_model)
            module.model = model
        args.default_roor_dir = Path(args.ckpt_path).parts[-4]
        name = Path(args.ckpt_path).parts[-3] + "_quantization"
        version = Path(args.ckpt_path).parts[-2]
    elif args.mode == "evaluate":
        if args.path_model:
            model = torch.load(args.path_model)
            module.model = model
        args.default_roor_dir = Path(args.ckpt_path).parts[-4]
        name = Path(args.ckpt_path).parts[-3] + "_evaluate"
        version = Path(args.ckpt_path).parts[-2]
        args.pruning_cfgs = None
    logger_tesnsorboard = TensorBoardLogger(save_dir=args.default_root_dir, name=name, version=version, log_graph=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=logger_tesnsorboard.log_dir,
            filename="{epoch}-{val_acc:.2f}",
            mode="max",
            monitor="val_acc",
            save_last=True,
        ),
        LearningRateMonitor("epoch"),
        ModelSummary(max_depth=1),
    ]
    # dataset
    dataset_train = RegressionPointDataset(args.path_data_train, args.dataset_cfgs,transforms="train")
    dataset_train_test = RegressionPointDataset(args.path_data_train, args.dataset_cfgs,transforms="train")
    dataset_val = RegressionPointDataset(args.path_data_train, args.dataset_cfgs,transforms="test")
    dataset_test = RegressionPointDataset(args.path_data_train, args.dataset_cfgs,transforms="test")

    loader_train = DataLoader(
        dataset=dataset_train,
        shuffle=True,
        collate_fn=dataset_train.get_collate_fn_for_dataloader(),
        **args.dataloader_cfgs,
    )
    loader_train_test = DataLoader(
        dataset=dataset_train_test,
        shuffle=False,
        collate_fn=dataset_train.get_collate_fn_for_dataloader(),
        **args.dataloader_cfgs,
    )
    loader_val = DataLoader(
        dataset=dataset_val,
        shuffle=False,
        collate_fn=dataset_val.get_collate_fn_for_dataloader(),
        **args.dataloader_cfgs,
    )
    loader_test = DataLoader(
        dataset=dataset_test,
        shuffle=False,
        collate_fn=dataset_val.get_collate_fn_for_dataloader(),
        **args.dataloader_cfgs,
    )
    profiler = SimpleProfiler(logger_tesnsorboard.log_dir)
    # trainer
    trainer = L.Trainer(logger=logger_tesnsorboard, callbacks=callbacks, profiler=profiler, **args.trainer)
    if args.mode == "train":
        trainer.fit(module, loader_train, loader_val)
    elif args.mode == "train_resume":
        trainer.fit(module, loader_train, loader_val, ckpt_path=args.ckpt_path)
    elif args.mode == "train_pruning":
        pruner = module.get_pruner(**args.pruning_cfgs)
        base_macs, base_nparams = tp.utils.count_ops_and_params(
            module.model, module.example_input_array.to(module.device)
        )
        for i in range(args.pruning_cfgs["iterative_steps"]):
            pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(module.model, module.example_input_array.to(module.device))
            print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
            trainer = L.Trainer(logger=logger, callbacks=callbacks, **args.trainer)
            trainer.fit(
                module,
                loader_train,
                loader_val,
            )
    elif args.mode == "train_quantization":
        pass
    if args.save_pt:
        path_pt = os.path.join(logger_tesnsorboard.log_dir, "best.pth")
        torch.save(module.model, path_pt)
    trainer = trainer = L.Trainer(**args.trainer)
    res_train = trainer.test(module, loader_train_test)
    res_val = trainer.test(module, loader_val)
    res_test = trainer.test(module, loader_test)

    # logger
    logger = logging.getLogger("trainer_classify.py")
    if not os.path.exists(logger_tesnsorboard.log_dir):
        os.makedirs(logger_tesnsorboard.log_dir)
    mode = "a" if os.path.exists(logger_tesnsorboard.log_dir + "/train.log") else "w"
    handler_file = logging.FileHandler(logger_tesnsorboard.log_dir + "/train.log", mode=mode)
    formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s:%(message)s")
    handler_file.setFormatter(formatter)
    handler_file.setLevel(logging.INFO)
    logger.addHandler(handler_file)
    logger.info("\n" + textwrap.fill(str(args), width=110))
    logger.info("save to : {}".format(logger_tesnsorboard.log_dir))
    logger.info("------------------------------dataset------------------------------")
    logger.info("train dataset : {}".format(len(dataset_train)))
    logger.info("val   dataset : {}".format(len(dataset_val)))
    logger.info("test  dataset : {}".format(len(dataset_test)))
    temp_path = Path(args.path_data_train)
    shutil.copyfile(str(temp_path), logger_tesnsorboard.log_dir + "/train_" + temp_path.name)
    temp_path = Path(args.path_data_val)
    shutil.copyfile(str(temp_path), logger_tesnsorboard.log_dir + "/val_" + temp_path.name)
    temp_path = Path(args.path_data_test)
    shutil.copyfile(str(temp_path), logger_tesnsorboard.log_dir + "/test_" + temp_path.name)
    logger.info("------------------------------module------------------------------")
    logger.info("\n" + str(torchinfo.summary(module.model, device="cpu", input_size=module.input_size)))
    logger.info("------------------------------test report------------------------------")
    logger.info("train : {}".format(res_train[0]))
    logger.info("val   : {}".format(res_val[0]))
    logger.info("test  : {}".format(res_test[0]))
    return


def cmd_inference(args):
    pass


def get_args():
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument(
        "--ckpt-path",
        default=None,
    )
    parser.add_argument(
        "--path-model",
        default=None,
    )
    parser.add_argument(
        "--module-cfgs",
        type=dict,
        default={
            "model_name": "DMPR",
            "model_hparams": {"input_size": (1, 3, 512, 512), "feature_map_channel": 6, "depth_factor": 32},
            "optimizer_name": "Adam",
            "optimizer_hparams": {"lr": 1e-3, "weight_decay": 1e-4},
        },
        help="Configs of module",
    )
    subparsers = parser.add_subparsers(required=True)
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("mode", choices=["train", "train_resume", "train_pruning", "train_quantization"])
    parser_train.add_argument("--default_root_dir", default="./logs")
    parser_train.add_argument("--save-pt", default=True)
    parser_train.add_argument("--save-onnx", default=True)
    parser_train.add_argument("--save-dir", default=True)
    parser_train.add_argument(
        "--path-data-train",
        default=r"database_sample/parking_slot/train.txt",
        help="Path to dataset for train (default: None)",
    )
    parser_train.add_argument(
        "--path-data-val",
        default=r"database_sample/parking_slot/train.txt",
        help="Path to dataset for val (default: None)",
    )
    parser_train.add_argument(
        "--path-data-test",
        default=r"database_sample/parking_slot/train.txt",
        help="Path to dataset for val (default: None)",
    )
    parser_train.add_argument(
        "--dataset-cfgs",
        type=dict,
        default={"class_to_index": {"city": 0, "highway": 1}},
        help="Path to the configuration file.if the path exists, ignore other parameters (default: None)",
    )
    parser_train.add_argument(
        "--dataloader-cfgs",
        type=dict,
        default={"batch_size": 32, "pin_memory": False, "num_workers": 4},
        help="Configs of dataloader for train",
    )
    parser_train.add_argument(
        "--trainer",
        type=dict,
        default={
            "devices": 1,
            "accelerator": "auto",
            "max_epochs": 5,
            "min_epochs": 1,
            "check_val_every_n_epoch": 1,
            "precision": "16-mixed",
            "log_every_n_steps": 16,
            "accumulate_grad_batches": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
        },
    )
    parser_train.add_argument(
        "--pruning-cfgs",
        type=dict,
        default={"pruning_ratio": 0.5, "iterative_steps": 1},
    )
    parser_train.set_defaults(func=cmd_train)
    # inference
    parser_inference = subparsers.add_parser("inference")
    parser_inference.add_argument(
        "mode", choices=["inference_in_image", "inference_on_image_with_label", "inference_on_video"]
    )
    parser_inference.add_argument("--dir-save", default=None)
    parser_inference.add_argument("--dir-image", default=None)
    parser_inference.add_argument("--path_data_txt", default=None)
    parser_inference.add_argument("--dir-video", default=None)
    parser_inference.set_defaults(func=cmd_inference)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.func(args)
