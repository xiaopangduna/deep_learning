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

from src.modules.trainer_classify import ClassifyModule

def cmd_train(args):
    module = ClassifyModule(**args.module_cfgs)
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
    logger_tesnsorboard = TensorBoardLogger(
        save_dir=args.default_root_dir, name=name, version=version, log_graph=True
    )
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
    dataset_train = ClassifyDataset(args.path_data_train, "train", args.dataset_cfgs)
    dataset_train_test = ClassifyDataset(args.path_data_train, "test", args.dataset_cfgs)
    dataset_val = ClassifyDataset(args.path_data_val, "val", args.dataset_cfgs)
    dataset_test = ClassifyDataset(args.path_data_test, "test", args.dataset_cfgs)
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
        pruner = module.set_pruner(**args.pruning_cfgs)
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
        default="logs/mobilenet_v3_small/version_0/last.ckpt",
    )
    parser.add_argument(
        "--path-model",
        default=None,
    )
    parser.add_argument(
        "--module-cfgs",
        type=dict,
        default={
            "model_name": "mobilenet_v3_small",
            "model_hparams": {"num_classes": 2, "input_size": (1, 3, 224, 224)},
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
        default=r"/home/xiaopangdun/project/image_classification_example/dataset_sample/train.txt",
        help="Path to dataset for train (default: None)",
    )
    parser_train.add_argument(
        "--path-data-val",
        default=r"/home/xiaopangdun/project/image_classification_example/dataset_sample/train.txt",
        help="Path to dataset for val (default: None)",
    )
    parser_train.add_argument(
        "--path-data-test",
        default=r"/home/xiaopangdun/project/image_classification_example/dataset_sample/train.txt",
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
            "limit_train_batches": 0.1,
            "limit_val_batches": 0.1,
            "limit_test_batches": 0.1,
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