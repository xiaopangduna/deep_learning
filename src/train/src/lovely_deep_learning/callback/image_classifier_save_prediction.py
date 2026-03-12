import os
import torch
from PIL import Image
import numpy as np
import lightning.pytorch as pl


class SavePredictionCallback(pl.Callback):
    """
    用于保存图像分类模型预测结果的回调函数
    """
    def __init__(self, save_dir="predictions"):
        """
        Args:
            output_dir: 保存预测结果的目录
        """
        self.output_dir = save_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
        # 提取图像和可能的标签
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        # 处理预测输出
        if isinstance(outputs, dict):
            # 从字典中提取预测结果
            predictions = outputs['predictions']
            probabilities = outputs['probabilities']
        else:
            # 假设输出就是预测结果
            predictions = outputs
            probabilities = None

        # 保存当前批次的图像和预测结果
        for idx, (image, pred) in enumerate(zip(images, predictions)):
            # 获取预测类别
            pred_class = pred.item() if hasattr(pred, 'item') else int(pred)
            
            # 生成文件名
            filename = f"batch_{batch_idx}_item_{idx}_pred_{pred_class}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # 转换图像格式并保存
            # 反标准化（如果需要）- 这里假设图像已经经过了标准化
            image_display = image.cpu().detach()
            
            # 将值限制在[0,1]范围内
            image_display = torch.clamp(image_display, 0, 1)
            
            # 转换为HWC格式用于保存
            if image_display.dim() == 3:
                image_display = image_display.permute(1, 2, 0).numpy()
            else:
                image_display = image_display.numpy()
            
            # 使用PIL保存图像
            # 将tensor值域从[0,1]转换到[0,255]，并转换为uint8类型
            image_pil = Image.fromarray((image_display * 255).astype(np.uint8))
            image_pil.save(filepath)