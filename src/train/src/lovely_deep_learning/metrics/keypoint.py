# -*- encoding: utf-8 -*-
"""
@File    :   keypoint.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/09/13 23:16:56
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
from typing import Optional
import torch
from torch import Tensor
from torchmetrics import Metric
import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment


class EuclideanDistance(Metric):
    # 接收两个由点组成的tensor，返回位置误差
    # Set to True if the metric is differentiable else set to False
    is_differentiable: Optional[bool] = None

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: Optional[bool] = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    # 位置，xy，方向，西塔，顺序
    def __init__(
        self,
        threshold_distance: float = 2.0,
        threshold_direction: float = 5.0,
        format: str = "xy",
        is_order: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold_distance = threshold_distance
        self.threshold_direction = threshold_direction
        self.format = format
        self.is_order = is_order
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("miss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("error", default=torch.tensor(0), dist_reduce_fx="sum")
        # pixl
        self.add_state("distance_error", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        # °
        self.add_state("direction_error", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, pred: ndarray, target: ndarray) -> None:
        # [[[x,y,a],[x,y,a]],[[x,y,a],[x,y,a]]]
        # mei you preds
        # mei you target

        if pred.size == 0 or target.size == 0:
            row_index = np.array([])
        else:
            distance_cost = self._get_distance_cost(pred[:, 0:2], target[:, 0:2])
            row_index, col_index = linear_sum_assignment(distance_cost)
            row_index, col_index = self._filter_index_by_threshold(
                distance_cost, row_index, col_index, self.threshold_distance, "min"
            )
            if self.format == "xya":
                # °
                direction_cost = self._get_direction_cost(pred[:, 2:3], target[:, 2:3])
                row_index, col_index = self._filter_index_by_threshold(
                    direction_cost, row_index, col_index, self.threshold_direction, "min"
                )
                if row_index.size != 0:
                    self.direction_error += torch.tensor(direction_cost[row_index, col_index].sum())
            if row_index.size != 0:
                self.distance_error += torch.tensor(distance_cost[row_index, col_index].sum(), dtype=torch.float32)

        self.total += torch.tensor(pred.shape[0])
        self.correct += torch.tensor(len(row_index))
        self.miss += torch.tensor(target.shape[0] - len(row_index))
        self.error += torch.tensor(pred.shape[0] - len(row_index))

    def compute(self) -> Tensor:

        res = {}
        # res["distance_error"] = self.distance_error / self.total
        res["distance_error"] = self._safe_divide(self.distance_error, self.correct)
        res["precision"] = self._safe_divide(self.correct, self.total)
        res["recall"] = self._safe_divide(self.correct, self.correct + self.miss)
        res["total"] = self.total
        res["correct"] = self.correct
        res["miss"] = self.miss
        res["error"] = self.error
        return res

    @staticmethod
    def _get_distance_cost(x: ndarray, y: ndarray):
        x_norm = (x * x).sum(axis=1, keepdims=True)
        y_norm = (y * y).sum(axis=1)
        distance = x_norm + y_norm - 2 * np.dot(x, y.T)
        return np.sqrt(distance)

    @staticmethod
    def _get_direction_cost(x: ndarray, y: ndarray):
        return np.abs(x - y.T)

    @staticmethod
    def _filter_index_by_threshold(cost, row_index, col_index, threshold, mode):
        res_row, res_col = [], []
        for i in range(len(row_index)):
            if cost[row_index[i]][col_index[i]] <= threshold:
                res_row.append(row_index[i])
                res_col.append(col_index[i])
        return np.array(res_row), np.array(res_col)

    @staticmethod
    def _safe_divide(num: Tensor, denom: Tensor) -> Tensor:
        denom[denom == 0.0] = 1
        num = num if num.is_floating_point() else num.float()
        denom = denom if denom.is_floating_point() else denom.float()
        return num / denom


# """
# Implementation of (mean) Average Precision metric for 2D keypoint detection.
# """

# from __future__ import annotations  # allow typing of own class objects

# import copy
# import math
# from dataclasses import dataclass
# from typing import Callable, Dict, List, Tuple

# import torch
# from torchmetrics import Metric
# from torchmetrics.utilities import check_forward_full_state_property


# @dataclass
# class Keypoint:
#     """A simple class datastructure for Keypoints,
#     dataclass is chosen over named tuple because this class is inherited by other classes
#     """

#     u: int
#     v: int

#     def l2_distance(self, keypoint: Keypoint):
#         return math.sqrt((self.u - keypoint.u) ** 2 + (self.v - keypoint.v) ** 2)


# @dataclass
# class DetectedKeypoint(Keypoint):
#     probability: float


# @dataclass(unsafe_hash=True)
# class ClassifiedKeypoint(DetectedKeypoint):
#     """
#     DataClass for a classified keypoint, where classified means determining if the detection is a True Positive of False positive,
#      with the given treshold distance and the gt keypoints from the frame

#     a hash is required for torch metric
#     cf https://github.com/PyTorchLightning/metrics/blob/2c8e46f87cb67186bff2c7b94bf1ec37486873d4/torchmetrics/metric.py#L570
#     unsafe_hash -> dirty fix to allow for hash w/o explictly telling python the object is immutable.
#     """

#     threshold_distance: int
#     true_positive: bool


# def keypoint_classification(
#     detected_keypoints: List[DetectedKeypoint],
#     ground_truth_keypoints: List[Keypoint],
#     threshold_distance: int,
# ) -> List[ClassifiedKeypoint]:
#     """Classifies keypoints of a **single** frame in True Positives or False Positives by searching for unused gt keypoints in prediction probability order
#     that are within distance d of the detected keypoint (greedy matching).

#     Args:
#         detected_keypoints (List[DetectedKeypoint]): The detected keypoints in the frame
#         ground_truth_keypoints (List[Keypoint]): The ground truth keypoints of a frame
#         threshold_distance: maximal distance in pixel coordinate space between detected keypoint and ground truth keypoint to be considered a TP

#     Returns:
#         List[ClassifiedKeypoint]: Keypoints with TP label.
#     """
#     classified_keypoints: List[ClassifiedKeypoint] = []

#     ground_truth_keypoints = copy.deepcopy(
#         ground_truth_keypoints
#     )  # make deep copy to do local removals (pass-by-reference..)

#     for detected_keypoint in sorted(detected_keypoints, key=lambda x: x.probability, reverse=True):
#         matched = False
#         for gt_keypoint in ground_truth_keypoints:
#             distance = detected_keypoint.l2_distance(gt_keypoint)
#             # add small epsilon to avoid numerical errors
#             if distance <= threshold_distance + 1e-5:
#                 classified_keypoint = ClassifiedKeypoint(
#                     detected_keypoint.u,
#                     detected_keypoint.v,
#                     detected_keypoint.probability,
#                     threshold_distance,
#                     True,
#                 )
#                 matched = True
#                 # remove keypoint from gt to avoid muliple matching
#                 ground_truth_keypoints.remove(gt_keypoint)
#                 break
#         if not matched:
#             classified_keypoint = ClassifiedKeypoint(
#                 detected_keypoint.u,
#                 detected_keypoint.v,
#                 detected_keypoint.probability,
#                 threshold_distance,
#                 False,
#             )
#         classified_keypoints.append(classified_keypoint)

#     return classified_keypoints


# def calculate_precision_recall(
#     classified_keypoints: List[ClassifiedKeypoint], total_ground_truth_keypoints: int
# ) -> Tuple[List[float], List[float]]:
#     """Calculates precision recall points on the curve for the given keypoints by varying the treshold probability to all detected keypoints
#      (i.e. by always taking one additional keypoint als a predicted event)

#     Note that this function is tailored towards a Detector, not a Classifier. For classifiers, the outputs contain both TP, FP and FN. Whereas for a Detector the
#     outputs only define the TP and the FP; the FN are not contained in the output as the point is exactly that the detector did not detect this event.

#     A detector is a ROI finder + classifier and the ROI finder could miss certain regions, which results in FNs that are hence never passed to the classifier.

#     This also explains why the scikit average_precision function states it is for Classification tasks only. Since it takes "total_gt_events" to be the # of positive_class labels.
#     The function can however be used by using as label (TP = 1, FP = 0) and by then multiplying the result with TP/(TP + FN) since the recall values are then corrected
#     to take the unseen events (FN's) into account as well. They do not matter for precision calcultations.
#     Args:
#         classified_keypoints (List[ClassifiedKeypoint]):
#         total_ground_truth_keypoints (int):

#     Returns:
#         Tuple[List[float], List[float]]: precision, recall entries. First entry is (1,0); last entry is (0,1).
#     """
#     precision = [1.0]
#     recall = [0.0]

#     true_positives = 0
#     false_positives = 0

#     for keypoint in sorted(classified_keypoints, key=lambda x: x.probability, reverse=True):
#         if keypoint.true_positive:
#             true_positives += 1
#         else:
#             false_positives += 1

#         precision.append(_zero_aware_division(true_positives, (true_positives + false_positives)))
#         recall.append(_zero_aware_division(true_positives, total_ground_truth_keypoints))

#     precision.append(0.0)
#     recall.append(1.0)
#     return precision, recall


# def calculate_ap_from_pr(precision: List[float], recall: List[float]) -> float:
#     """Calculates the Average Precision using the AUC definition (COCO-style)

#     # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
#     # AUC AP.

#     Args:
#         precision (List[float]):
#         recall (List[float]):

#     Returns:
#         (float): average precision (between 0 and 1)
#     """

#     smoothened_precision = copy.deepcopy(precision)

#     for i in range(len(smoothened_precision) - 2, 0, -1):
#         smoothened_precision[i] = max(smoothened_precision[i], smoothened_precision[i + 1])

#     ap = 0
#     for i in range(len(recall) - 1):
#         ap += (recall[i + 1] - recall[i]) * smoothened_precision[i + 1]

#     return ap


# class KeypointAPMetric(Metric):
#     """torchmetrics-like interface for the Average Precision implementation"""

#     full_state_update = False

#     def __init__(self, keypoint_threshold_distance: float, dist_sync_on_step=False):
#         """

#         Args:
#             keypoint_threshold_distance (float): distance from ground_truth keypoint that is used to classify keypoint as TP or FP.
#         """

#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         self.keypoint_threshold_distance = keypoint_threshold_distance

#         default: Callable = lambda: []
#         self.add_state("classified_keypoints", default=default(), dist_reduce_fx="cat")  # list of ClassifiedKeypoints
#         self.add_state("total_ground_truth_keypoints", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, detected_keypoints: List[DetectedKeypoint], gt_keypoints: List[Keypoint]):

#         classified_img_keypoints = keypoint_classification(
#             detected_keypoints, gt_keypoints, self.keypoint_threshold_distance
#         )

#         self.classified_keypoints += classified_img_keypoints

#         self.total_ground_truth_keypoints += len(gt_keypoints)

#     def compute(self):
#         p, r = calculate_precision_recall(self.classified_keypoints, int(self.total_ground_truth_keypoints.cpu()))
#         m_ap = calculate_ap_from_pr(p, r)
#         return m_ap


# class KeypointAPMetrics(Metric):
#     """
#     Torchmetrics-like interface for calculating average precisions over different keypoint_threshold_distances.
#     Uses KeypointAPMetric class.
#     """

#     full_state_update = False

#     def __init__(self, keypoint_threshold_distances: List[int], dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         self.ap_metrics = [KeypointAPMetric(dst, dist_sync_on_step) for dst in keypoint_threshold_distances]

#     def update(self, detected_keypoints: List[DetectedKeypoint], gt_keypoints: List[Keypoint]):
#         for metric in self.ap_metrics:
#             metric.update(detected_keypoints, gt_keypoints)

#     def compute(self) -> Dict[float, float]:
#         result_dict = {}
#         for metric in self.ap_metrics:
#             result_dict.update({metric.keypoint_threshold_distance: metric.compute()})
#         return result_dict

#     def reset(self) -> None:
#         for metric in self.ap_metrics:
#             metric.reset()


# def _zero_aware_division(num: float, denom: float) -> float:
#     if num == 0:
#         return 0
#     if denom == 0 and num != 0:
#         return float("inf")
#     else:
#         return num / denom


# if __name__ == "__main__":
#     print(
#         check_forward_full_state_property(
#             KeypointAPMetric,
#             init_args={"keypoint_threshold_distance": 2.0},
#             input_args={"detected_keypoints": [DetectedKeypoint(10, 20, 0.02)], "gt_keypoints": [Keypoint(10, 23)]},
#         )
#     )
