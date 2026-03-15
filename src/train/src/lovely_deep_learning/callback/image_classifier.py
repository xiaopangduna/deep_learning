import os

import lightning.pytorch as pl
import cv2
from pathlib import Path
import pandas as pd

from ..dataset.image_classifier import ImageClassifierDataset


class ImageClassifierCallback(pl.Callback):
    """
    用于保存图像分类模型预测结果的回调函数
    """

    def __init__(self, save_dir: str = None, test_only_save_mistake=True):
        """
        Args:
            output_dir: 保存预测结果的目录
        """
        super().__init__()
        self.test_only_save_mistake = test_only_save_mistake
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir_test = self.save_dir / "test"
            self.save_dir_pred = self.save_dir / "predict"
            self.csv_table_test = []
            self.csv_table_pred = []  # 初始化为空列表，用于存储预测结果

            os.makedirs(self.save_dir_test, exist_ok=True)
            os.makedirs(self.save_dir_pred, exist_ok=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.save_dir is None:
            return
        dataset: ImageClassifierDataset = trainer.test_dataloaders.dataset

        img_tv = batch[0]["img_tv_transformed"]  # [B,3,H,W]

        class_id_pred = outputs["class_id_pred"]
        class_id_conf = outputs["class_id_conf"]

        img_tv = img_tv.detach().cpu()

        B = img_tv.shape[0]

        for i in range(B):
            img_path = Path(batch[0]["img_path"][i])
            img = img_tv[i]  # [3,H,W]

            cur_class_id = batch[1]["class_id"][i].item()
            cur_class_name = batch[1]["class_name"][i]

            cur_class_id_pred = class_id_pred[i].item()
            cur_class_name_pred = dataset.map_class_id_to_class_name[cur_class_id_pred]
            cur_confidence_pred = class_id_conf[i].item()

            img_np = dataset.convert_img_from_tensor_to_numpy_uint8(img)
            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np,
                class_name=cur_class_name,
                class_id=cur_class_id,
                class_name_pred=cur_class_name_pred,
                class_id_pred=cur_class_id_pred,
                class_id_conf=cur_confidence_pred,
            )

            save_path = self.save_dir_test / (img_path.stem + ".jpg")
            cv2.imwrite(save_path, img_np)

            self.csv_table_test.append(
                {
                    "img_path": str(img_path),
                    "class_id": cur_class_id,
                    "class_id_pred": cur_class_id_pred,
                    "class_name_pred": cur_class_name_pred,
                    "class_name": cur_class_name,
                    "confidence_pred": cur_confidence_pred,
                    "save_path": str(save_path),
                }
            )

    def on_test_epoch_start(self, trainer, pl_module):
        self.csv_table_test = []
        return super().on_test_epoch_start(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        if self.csv_table_test:
            df = pd.DataFrame(self.csv_table_test)
            csv_save_path = self.save_dir / "test_results.csv"
            df.to_csv(csv_save_path, index=False)
            print(f"预测结果已保存到: {csv_save_path}")

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        """
        每个预测批次结束时保存图像和预测结果
        """
        if self.save_dir is None:
            return
        dataset: ImageClassifierDataset = trainer.predict_dataloaders.dataset

        img_tv = batch[0]["img_tv_transformed"]  # [B,3,H,W]

        class_id_pred = outputs["class_id_pred"]
        class_id_conf = outputs["class_id_conf"]

        img_tv = img_tv.detach().cpu()

        B = img_tv.shape[0]

        for i in range(B):
            img_path = Path(batch[0]["img_path"][i])
            img = img_tv[i]  # [3,H,W]
            cur_class_id_pred = class_id_pred[i].item()
            cur_class_name_pred = dataset.map_class_id_to_class_name[cur_class_id_pred]
            cur_confidence_pred = class_id_conf[i].item()

            img_np = dataset.convert_img_from_tensor_to_numpy_uint8(img)
            img_np = dataset.draw_target_and_predict_label_on_numpy(
                img_np, class_name_pred=cur_class_name_pred, class_id_pred=cur_class_id_pred, class_id_conf=cur_confidence_pred
            )

            save_path = self.save_dir_pred / (img_path.stem + ".jpg")
            cv2.imwrite(save_path, img_np)

            self.csv_table_pred.append(
                {
                    "img_path": str(img_path),
                    "class_id_pred": cur_class_id_pred,
                    "class_name_pred": cur_class_name_pred,
                    "confidence_pred": cur_confidence_pred,
                    "save_path": str(save_path),
                }
            )

    def on_predict_epoch_start(self, trainer, pl_module):
        self.csv_table_pred = []
        return super().on_predict_epoch_start(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        if self.csv_table_pred:
            df = pd.DataFrame(self.csv_table_pred)
            csv_save_path = self.save_dir / "prediction_results.csv"
            df.to_csv(csv_save_path, index=False)
            print(f"预测结果已保存到: {csv_save_path}")
