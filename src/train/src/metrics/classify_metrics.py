import torch
import numpy as np
import warnings
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from src.metrics.base_metrics import BaseMetrics


warnings.filterwarnings("ignore")
__all__ = ["ClassifyMetrics"]


class ClassifyMetrics(BaseMetrics):
    def __init__(self, cfgs: dict):
        self.class_to_index = cfgs["class_to_index"]
        self.classes = [0] * len(self.class_to_index)
        for key in self.class_to_index.keys():
            self.classes[self.class_to_index[key]] = key
        self.predicted_value = None
        self.target_value = None
        self.metrics = {"accuracy": -1, "precision": -1, "recall": -1, "f1_score": -1}
        self.metric = None
        self.metric_best = None
        return

    def update_metrics_batch(self, outputs: torch.Tensor, targets: torch.Tensor):
        _, index_outputs = torch.max(outputs, dim=1)
        _, index_targets = torch.max(targets, dim=1)
        if self.predicted_value == None:
            self.predicted_value = index_outputs
        else:
            self.predicted_value = torch.cat((self.predicted_value, index_outputs), dim=0)
        if self.target_value == None:
            self.target_value = index_targets
        else:
            self.target_value = torch.cat((self.target_value, index_targets), dim=0)
        return

    def update_metrics_epoch(self):
        predicted_value = self.predicted_value.cpu().numpy()
        target_value = self.target_value.cpu().numpy()
        acc = accuracy_score(target_value, predicted_value)
        pre = precision_score(target_value, predicted_value, average="micro", zero_division=0)
        f1 = f1_score(target_value, predicted_value, average="micro", zero_division=0)
        recall = recall_score(target_value, predicted_value, average="micro", zero_division=0)
        self.metrics["accuracy"] = acc
        self.metrics["precision"] = pre
        self.metrics["f1_score"] = f1
        self.metrics["recall"] = recall
        self.metric = self.metrics["f1_score"]
        return

    def get_report(self):
        predicted_value = self.predicted_value.cpu().numpy()
        target_value = self.target_value.cpu().numpy()
        report = classification_report(target_value, predicted_value, target_names=self.classes)
        return "\n" + report

    def reset_metrics(self):
        self.predicted_value = None
        self.target_value = None
        return

    # 拼接numpy
    # 返回字符串格式的信息
