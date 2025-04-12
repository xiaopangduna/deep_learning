# -*- encoding: utf-8 -*-
"""
@File    :   test_trainer_keypoint.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/09/13 22:51:47
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pytest
from torch.utils.data import DataLoader
import lightning as L
from src.modules.object_detection import DirectionalCornerDetectionModule
from src.datasets.object_detection import DirectionalCornerDetectionDataset


class TestDirectionalCornerDetectionModule(object):
    def test_get_model(self):
        model_name = "DMPR"
        model_hparams = {"input_size": (1, 3, 512, 512), "feature_map_channel": 6, "depth_factor": 32}
        optimizer_name = "adam"
        optimizer_hparams = {}
        module = DirectionalCornerDetectionModule(model_name, model_hparams, optimizer_name, optimizer_hparams)
        pass

    def test_train(self):
        model_name = "DMPR"
        model_hparams = {"input_size": (1, 3, 512, 512), "feature_map_channel": 7, "depth_factor": 32}
        optimizer_name = "Adam"
        optimizer_hparams = {}
        module = DirectionalCornerDetectionModule(model_name, model_hparams, optimizer_name, optimizer_hparams)

        path_txt = ["../database/ps2.0/train.txt"]
        cfgs = {"output_size": [16, 16], "input_size": [3, 512, 512], "classes": ["T","L"]}
        indexs_annotations = ("data_image", "label_0")
        transforms = "train"  # "train", "val", "test" "None"
        dataset = DirectionalCornerDetectionDataset(path_txt, cfgs, indexs_annotations, transforms=transforms)
        dataloader = DataLoader(
            dataset=dataset, shuffle=True, collate_fn=dataset.get_collate_fn_for_dataloader(), batch_size=8
        )
        trainer = L.Trainer(max_epochs=2,limit_train_batches=3,limit_val_batches=3)
        # trainer.fit(model=module, train_dataloaders=dataloader, val_dataloaders=dataloader)
        trainer.validate(model=module, dataloaders=dataloader)
        pass


if __name__ == "__main__":
    temp_class = TestDirectionalCornerDetectionModule()
    # temp_class.test_get_model()
    temp_class.test_train()
