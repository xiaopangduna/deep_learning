import lightning.pytorch as pl
import torch
import torch.nn as nn

import torch.nn.functional as F
from lightning.pytorch.cli import instantiate_class
from typing import Any

from torchvision.utils import make_grid
from lovely_deep_learning.model.DAGNet import DAGNet
from ..dataset.image_classifier import ImageClassifierDataset

class ImageClassifierModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        model=None,
        init_type: str | None = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
    ):
        """
        lr: 学习率

        """
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.optimizer_cfg = optimizer
        self.lr_scheduler_cfg = lr_scheduler
        if self.optimizer_cfg is None:
            raise ValueError("`optimizer` config is required in YAML (model.init_args.optimizer).")
        if self.lr_scheduler_cfg is None:
            raise ValueError("`lr_scheduler` config is required in YAML (model.init_args.lr_scheduler).")
        self.model = DAGNet(**model)
        self.example_input_array = torch.randn(1, 3, 224, 224)

        self._graph_logged = False

        # 根据 init_type 对模型进行自定义初始化；为 None 时使用默认初始化
        # if init_type is not None:
        #     self._initialize_weights(init_type)

    def forward(self, x):
        return self.model([x])

    def training_step(self, batch, batch_idx):
        net_in, net_out = batch
        img: torch.Tensor = net_in["img_tv_transformed"]
        batch_size = img.shape[0]
        class_id = net_out["class_id"]
        logits = self(img)
        loss = F.cross_entropy(logits[0], class_id)
        probabilities = F.softmax(logits[0], dim=1)
        class_id_conf = probabilities.max(dim=1)[0]
        preds = logits[0].argmax(dim=1)
        acc = (preds == class_id).float().mean()

        if batch_idx == 0:
            try:
                dataset: ImageClassifierDataset = self.trainer.datamodule.train_dataset
                self._log_images_with_target_and_predictions(img, net_out, preds, class_id_conf, dataset, "train")
            except Exception as e:
                print(f"Warning: failed to log images at step {self.global_step}, {e}")

        self.log("train_loss", loss, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        net_in, net_out = batch
        img = net_in["img_tv_transformed"]
        batch_size = img.shape[0]
        class_id = net_out["class_id"]
        logits = self(img)
        loss = F.cross_entropy(logits[0], class_id)
        preds = logits[0].argmax(dim=1)
        probabilities = F.softmax(logits[0], dim=1)
        class_id_conf = probabilities.max(dim=1)[0]
        acc = (preds == class_id).float().mean()

        if batch_idx == 0:
            try:
                dataset: ImageClassifierDataset = self.trainer.datamodule.val_dataset
                self._log_images_with_target_and_predictions(img, net_out, preds, class_id_conf, dataset, "val")
            except Exception as e:
                print(f"Warning: failed to log images at step {self.global_step}, {e}")

        self.log("val_loss", loss, batch_size=batch_size,on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, batch_size=batch_size,on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        net_in, net_out = batch
        img = net_in["img_tv_transformed"]
        class_id = net_out["class_id"]

        logits = self(img)
        loss = F.cross_entropy(logits[0], class_id)
        class_id_pred = logits[0].argmax(dim=1)
        acc = (class_id_pred == class_id).float().mean()

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        probabilities = F.softmax(logits[0], dim=1)
        class_id_conf = probabilities.max(dim=1)[0]

        return {"class_id_pred": class_id_pred, "class_id_conf": class_id_conf}

    def predict_step(self, batch, batch_idx):
        net_in, _ = batch
        img = net_in["img_tv_transformed"]
        logits = self(img)
        probabilities = F.softmax(logits[0], dim=1)
        class_id_conf = probabilities.max(dim=1)[0]
        class_id_pred = logits[0].argmax(dim=1)
        return {"class_id_pred": class_id_pred, "class_id_conf": class_id_conf}

    def configure_optimizers(self):
        optimizer = instantiate_class(self.parameters(), self.optimizer_cfg)
        scheduler = instantiate_class(optimizer, self.lr_scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _initialize_weights(self, init_type: str):
        """
        根据 init_type 对可训练层进行权重初始化。

        支持的 init_type 示例：
        - "kaiming_normal"
        - "kaiming_uniform"
        - "xavier_normal"
        - "xavier_uniform"
        - "normal"
        - "uniform"
        - "orthogonal"
        """
        init_type = init_type.lower()

        for m in self.model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                if init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                elif init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                elif init_type == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                elif init_type == "uniform":
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def on_fit_start(self):
        hparams = {"lr": 0.001, "batch_size": 32}
        metrics = {"val_loss": 0.12, "val_acc": 0.95}
        self.logger.experiment.add_hparams(hparams, metrics)
        if not self._graph_logged and self.logger is not None:
            device = next(self.model.parameters()).device  # 获取模型所在设备
            example_input = self.example_input_array.to(device)
            self.logger.experiment.add_graph(self, example_input)
            self._graph_logged = True

    def _log_images_with_target_and_predictions(self, img, net_out, preds, class_id_conf, dataset, log_prefix):
        """
        辅助函数：记录带有预测结果的图像到日志
        """
        imgs_with_label = []
        for i in range(min(9, img.shape[0])):
            img_tensor = img[i]
            class_name = net_out["class_name"][i]
            class_id = net_out["class_id"][i]
            class_id_pred = preds[i].item()
            class_name_pred = dataset.map_class_id_to_class_name[class_id_pred]
            confidence_pred = class_id_conf[i].item()

            img_np = dataset.convert_img_from_tensor_to_numpy(img_tensor)

            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np,
                class_name=class_name,
                class_id=class_id,
                class_name_pred=class_name_pred,
                class_id_pred=class_id_pred,
                class_id_conf=confidence_pred,
            )
            img_with_label_tensor = dataset.convert_img_from_numpy_to_tensor_uint8(img_np)
            imgs_with_label.append(img_with_label_tensor)
        img_grid = make_grid(imgs_with_label, nrow=3)
        self.logger.experiment.add_image(f"{log_prefix}/sample_batch", img_grid, global_step=self.global_step)

    def on_train_epoch_end(self):
        try:
            # 记录权重和梯度直方图
            self.logger.experiment.add_histogram("fc/weights", self.model.layers.fc.weight, self.global_step)
        except Exception as e:
            print(f"Warning: failed to log histogram at step {self.global_step}, {e}")
