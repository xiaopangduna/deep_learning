import os
import json
from typing import Dict, Optional, Tuple, Union, Any, List
from pathlib import Path
import argparse
import sys
import logging
import textwrap
import shutil
import lightning.pytorch
import yaml
import lightning as L
import torch
from torch import nn, optim
import torchvision
import torchinfo
import lightning
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, ModelSummary

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from lightning.pytorch.loops.fit_loop import _FitLoop
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
from lightning.pytorch.cli import LightningCLI
import src
from src.modules.trainer_classify import ClassifyModule, ClassifyDataset
import src.datasets as my_datasets
from src import modules
from src.utils.profiler_pytorch import compare_models
import logging
logging.basicConfig(level=logging.INFO)
# PrintTableMetricsCallback：在每个epoch结束后打印一份结果整理表格。


def get_datasets(args):
    datasets = {}
    dataset_class = getattr(my_datasets, args.dataset_class)
    for dataset_key in args.datasets:
        dataset = dataset_class(**args.datasets[dataset_key])
        datasets[dataset_key] = dataset
    return datasets


def get_dataloaders(args, datasets):
    dataloaders = {}
    dataloader_class = getattr(torch.utils.data, args.dataloader_class)
    for dataloader_key in args.dataloaders:
        dataloader = dataloader_class(
            dataset=datasets[dataloader_key],
            collate_fn=datasets[dataloader_key].get_collate_fn_for_dataloader(),
            **args.dataloaders[dataloader_key],
        )
        dataloaders[dataloader_key] = dataloader
    return dataloaders


def get_callbacks(args):
    callbacks = []
    for callback_name in args.callbacks:
        callback_class = getattr(lightning.pytorch.callbacks, callback_name)
        callback = callback_class(**(getattr(args, callback_name)))
        callbacks.append(callback)
    return callbacks


def get_profiler(args):
    profiler_class = getattr(lightning.pytorch.profilers, args.profiler, None)
    profiler = profiler_class(**(getattr(args, args.profiler))) if profiler_class else None
    return profiler


def get_module(args):
    module_class = getattr(modules, args.module_class)
    module_cfgs = getattr(args, "module_cfgs", None)
    if module_cfgs:
        module = module_class(**args.module_cfgs)
    else:
        module = module_class.load_from_checkpoint(checkpoint_path=args.path_ckpt)
    return module


def get_loggers(args, mode):
    loggers = []
    if getattr(args, "path_ckpt", None):
        path_ckpt = Path(args.path_ckpt)
    else:
        path_ckpt = None
    if mode == "train":
        for logger_name in args.loggers:
            logger_class = getattr(lightning.pytorch.loggers, logger_name)
            logger = logger_class(**(getattr(args, logger_name)))
            loggers.append(logger)
    elif mode == "train_resume":
        for logger_name in args.loggers:
            if logger_name == "TensorBoardLogger":
                args.TensorBoardLogger["name"] = path_ckpt.parts[-4]
                args.TensorBoardLogger["version"] = path_ckpt.parts[-3]
            logger_class = getattr(lightning.pytorch.loggers, logger_name)
            logger = logger_class(**(getattr(args, logger_name)))
            loggers.append(logger)
    elif mode == "train_pruning":
        for logger_name in args.loggers:
            if logger_name == "TensorBoardLogger":
                pruning_mode = "single" if len(args.pruning_train_epochs) == 1 else "multiple"
                args.TensorBoardLogger["name"] = (
                    path_ckpt.parts[-4] + "_pruning_" + str(args.pruning_cfgs["pruning_ratio"]) + "_" + pruning_mode
                )
                # args.TensorBoardLogger["version"] = None
            logger_class = getattr(lightning.pytorch.loggers, logger_name)
            logger = logger_class(**(getattr(args, logger_name)))
            loggers.append(logger)
    elif mode == "train_quantization":
        for logger_name in args.loggers:
            if logger_name == "TensorBoardLogger":
                args.TensorBoardLogger["name"] = path_ckpt.parts[-4] + "_quantization"
                # args.TensorBoardLogger["version"] = None
            logger_class = getattr(lightning.pytorch.loggers, logger_name)
            logger = logger_class(**(getattr(args, logger_name)))
            loggers.append(logger)

    return loggers


def save_config_to_yaml(args: argparse.Namespace, path_save: str):
    """
    将配置保存为 YAML 文件。如果文件已存在，则自动在文件名后添加递增的数字。

    参数:
        args: 需要保存的配置（可以是字典、Namespace 对象等）。
        path_save (str 或 Path): 目标文件路径。
    """
    path = Path(path_save)
    parent_dir = path.parent
    stem = path.stem  # 文件名（不含后缀）
    suffix = path.suffix  # 文件后缀（如 .yaml）

    # 确保父目录存在
    parent_dir.mkdir(parents=True, exist_ok=True)

    # 如果文件已存在，则递增文件名
    counter = 1
    while path.exists():
        new_stem = f"{stem}_{counter}"
        path = parent_dir / f"{new_stem}{suffix}"
        counter += 1

    # 将配置保存为 YAML 文件
    with open(path, mode="w", encoding="utf-8") as f:
        yaml.dump(vars(args), f, allow_unicode=True)

    print(f"配置已保存到 {path}")
    return path  # 返回最终保存的文件路径


def cmd_train(args, mode: str):
    # Step : Set datasets
    datasets = get_datasets(args)
    # Step : Set Dataloaders
    dataloaders = get_dataloaders(args, datasets)
    # Step : get module
    module = get_module(args)
    # Step : Set callbacks
    callbacks = get_callbacks(args)
    # Step : Set profiler
    profiler = get_profiler(args)
    # Step : Set logger
    loggers = get_loggers(args, mode)
    # Step : Set trainer
    save_config_to_yaml(args, loggers[0].log_dir + "/config_" + mode + ".yaml")
        # 打印模型各层的权重均值和方差
    # for name, param in module.model.named_parameters():
    #     print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    if mode == "train":
        trainer = L.Trainer(logger=loggers, callbacks=callbacks, profiler=profiler, **args.trainer)
        trainer.fit(module, dataloaders["train"], dataloaders["val"])
    elif mode == "train_resume":
        trainer = L.Trainer(logger=loggers, callbacks=callbacks, profiler=profiler, **args.trainer)
        # automatically restores model, epoch, step, LR schedulers, etc...
        trainer.fit(module, dataloaders["train"], dataloaders["val"], ckpt_path=args.path_ckpt)
    elif mode == "train_pruning":
        args.pruning_cfgs["iterative_steps"] = len(args.pruning_train_epochs)
        module.set_pruner(**args.pruning_cfgs)
        base_macs, base_nparams = tp.utils.count_ops_and_params(
            module.model, module.example_input_array.to(module.device)
        )
        trainer = L.Trainer(logger=loggers, callbacks=callbacks, profiler=profiler, max_epochs=0, **args.trainer)
        ModelSummary.on_fit_start(trainer, module)
        for i in range(args.pruning_cfgs["iterative_steps"]):
            trainer.fit_loop.max_epochs += args.pruning_train_epochs[i]
            module.pruner.step()
            macs, nparams = tp.utils.count_ops_and_params(module.model, module.example_input_array.to(module.device))
            print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
            trainer.fit(
                module,
                dataloaders["train"],
                dataloaders["val"],
            )
        print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
    elif mode == "train_quantization":
        module.set_quantization_init()
        module_class = getattr(modules, args.module_class)
        module = module_class.load_from_checkpoint(checkpoint_path=args.path_ckpt)
        module.set_quantization_ptq(dataset=datasets["val"])
        trainer = L.Trainer(logger=loggers, callbacks=callbacks, profiler=profiler, **args.trainer)
        trainer.validate(module, dataloaders["val"])
        trainer.fit(module, dataloaders["train"], dataloaders["val"])
    return


def cmd_test(args, mode):
    # Step : Set datasets
    datasets = get_datasets(args)
    # Step : Set Dataloaders
    dataloaders = get_dataloaders(args, datasets)
    # Step : get module
    module_class = getattr(modules, args.module_class)
    try:
        module = module_class.load_from_checkpoint(checkpoint_path=args.path_ckpt)
    except:
        module = module_class(**args.module_cfgs)
        model = torch.load(args.path_model)
        module.model = model
    trainer = L.Trainer(**args.trainer)
    trainer.test(module, dataloaders["test"])
    # module.profile(**args.profile_cfgs)
    # try:
        # module.profile(**args.profile_cfgs)
    # except:
    #     print("faile to profile")
    return


def cmd_predict(args,mode):
    # Step : Set datasets
    datasets = get_datasets(args)
    # Step : Set Dataloaders
    dataloaders = get_dataloaders(args, datasets)
    # Step : get module
    module = get_module(args)
    trainer = L.Trainer(**args.trainer)
    trainer.predict(module,dataloaders["predict"],return_predictions=False)
    pass


def cmd_export_onnx(args, mode):
    model = torch.load(args.path_model)
    module_class = getattr(modules, args.module_class)
    if not args.export_onnx_cfgs["path_save"]:
        args.export_onnx_cfgs["path_save"] = str(Path(args.path_model).with_suffix(".onnx"))
    module_class.export(model, **args.export_onnx_cfgs)

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("parameters")
    parser.add_argument(
        "mode",
        choices=["train", "train_resume", "train_pruning", "train_quantization", "test", "predict", "export_onnx"],
    )
    parser.add_argument("--path-config", default="configs/directional_corner_detection.yml")
    args = parser.parse_args()
    with open(args.path_config, mode="r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    if args.mode == "train":
        cmd_train(argparse.Namespace(**configs[args.mode]), "train")
    elif args.mode == "train_resume":
        cmd_train(argparse.Namespace(**configs[args.mode]), "train_resume")
    elif args.mode == "train_pruning":
        cmd_train(argparse.Namespace(**configs[args.mode]), "train_pruning")
    elif args.mode == "train_quantization":
        cmd_train(argparse.Namespace(**configs[args.mode]), "train_quantization")
    elif args.mode.startswith("test"):
        cmd_test(argparse.Namespace(**configs[args.mode]), "test")
    elif args.mode.startswith("predict"):
        cmd_predict(argparse.Namespace(**configs[args.mode]), "predict")
    elif args.mode.startswith("export_onnx"):
        cmd_export_onnx(argparse.Namespace(**configs[args.mode]), "export_onnx")
    pass
    # 使用配置文件的命令行，不使用无配置的命令行

    #
    # # inference
    # parser_inference = subparsers.add_parser("inference")
    # parser_inference.add_argument(
    #     "mode", choices=["inference_in_image", "inference_on_image_with_label", "inference_on_video"]
    # )
    # parser_inference.add_argument("--dir-save", default=None)
    # parser_inference.add_argument("--dir-image", default=None)
    # parser_inference.add_argument("--path_data_txt", default=None)
    # parser_inference.add_argument("--dir-video", default=None)
    # parser_inference.set_defaults(func=cmd_inference)
