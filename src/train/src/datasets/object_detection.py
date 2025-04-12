# -*- encoding: utf-8 -*-
"""
@File    :   object_detection.py
@Python  :   python3.8
@version :   0.0
@Time    :   2025/03/03 21:45:17
@Author  :   xiaopangdun
@Email   :   18675381281@163.com
@Desc    :   This is a simple example
"""

import json
from pathlib import Path
import warnings
import math
import copy
import lightning as L
import torch
from torchvision.transforms import transforms as T

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import json
from scipy.io import loadmat
import os

from src.datasets.base_dataset import BaseDataset


class DirectionalCornerDetectionDataset(BaseDataset):

    def __getitem__(self, index):
        try:
            path_datas = self.path_datas[index]
            path_labels = self.path_labels[index]
            path_img = path_datas[0]
            path_keypoints = path_labels[0] if path_labels else None
            net_in, net_out = {}, {}
            img = cv2.imread(path_img)
            if img is None:
                raise FileNotFoundError(f"Image file not found: {path_img}")
            keypoint_raw = {
                "U_box": [],
                "X_box": [],
                "X": [],
                "K": [],
                "L": [],
                "T": [],
            }
            # keypoint_valid = copy.deepcopy(keypoint_raw)
            keypoint_raw, keypoint_valid = self._parse_labelme_json(path_keypoints, keypoint_raw)
            if self.transforms:
                if path_labels:
                    transformed = self.transforms(
                        image=img,
                        keypoints=keypoint_valid["L"],
                        keypoints1=keypoint_valid["T"],
                        keypoints2=keypoint_valid["X"],
                        keypoints3=keypoint_valid["K"],
                    )
                    img = transformed["image"]
                    keypoint_valid["L"] = transformed["keypoints"]
                    keypoint_valid["T"] = transformed["keypoints1"]
                    keypoint_valid["X"] = transformed["keypoints2"]
                    keypoint_valid["K"] = transformed["keypoints3"]

                else:
                    transformed = self.transforms(
                        image=img,
                    )
                    img = transformed["image"]
            del keypoint_valid["K"]
            del keypoint_valid["X"]
            keypoint_tensor = self.convert_valid_to_tensor(
                keypoint_valid, [img.shape[0], img.shape[1]], self.cfgs["output_size"]
            )
            net_in["img"] = img
            net_in["img_tensor"] = DirectionalCornerDetectionDataset.convert_image_from_numpy_to_tensor(img)
            net_in["img_path"] = path_img
            net_out["keypoint_raw"] = keypoint_raw if path_keypoints else None
            net_out["keypoint_valid"] = keypoint_valid if path_keypoints else None
            net_out["keypoint_tensor"] = keypoint_tensor if path_keypoints else None
            net_out["keypoint_path"] = path_keypoints if path_keypoints else None
            return net_in, net_out
        except IndexError:
            raise IndexError(f"Index {index} is out of range.")
        except FileNotFoundError as e:
            warnings.warn(str(e))
            return None, None
    @staticmethod
    def convert_image_from_numpy_to_tensor(img):
        return T.ToTensor()(img)
    @staticmethod
    def convert_image_from_tensor_to_numpy(img):
        if img.is_cuda:
            img = img.cpu()
        # 转换为 NumPy 数组
        numpy_array = img.numpy()
        # 调整维度顺序从 (C, H, W) 到 (H, W, C)
        numpy_array = numpy_array.transpose(1, 2, 0)

        # 将像素值从 [0.0, 1.0] 转换回 [0, 255]
        numpy_array = (numpy_array * 255).astype(np.uint8)
        # RGB to BGR
        numpy_array = numpy_array[:, :, ::-1]
        return numpy_array
    def _parse_labelme_json(self, path_labelme, keypoints_raw):
        try:
            with open(path_labelme, "r") as f:
                labelme = json.load(f)
            keypoints_raw = self.filter_shapes_from_labelme(labelme, keypoints_raw)
            keypoints_valid = self.convert_raw_to_valid(keypoints_raw, {"L": [], "T": []})
            return keypoints_raw, keypoints_valid
        except FileNotFoundError:
            warnings.warn(f"Label file not found: {path_labelme}")
            return keypoints_raw, {}

    def get_transforms_for_train(self):
        transforms = A.Compose(
            [
                # A.VerticalFlip(),
                # A.HorizontalFlip(),
                A.Rotate(),
                A.OneOf(
                    [
                        # A.HueSaturationValue(),
                        A.RGBShift(),
                        # A.RandomFog(),
                        A.RandomShadow(),
                        # A.RandomRain(),
                    ]
                ),
                A.Resize(self.cfgs["input_size"][1], self.cfgs["input_size"][2]),
            ],
            keypoint_params=A.KeypointParams(format="xya", remove_invisible=True),
            additional_targets={
                "keypoints1": "keypoints",
                "keypoints2": "keypoints",
                "keypoints3": "keypoints",
            },
        )
        return transforms

    def get_transforms_for_val(self):
        transforms = A.Compose(
            [
                A.Resize(self.cfgs["input_size"][1], self.cfgs["input_size"][2]),
            ],
            keypoint_params=A.KeypointParams(format="xya", remove_invisible=True),
            additional_targets={
                "keypoints1": "keypoints",
                "keypoints2": "keypoints",
                "keypoints3": "keypoints",
            },
        )
        return transforms

    @staticmethod
    def convert_raw_to_valid(shapes: dict, target: dict = {"L": [], "T": []}):
        L = []
        for box in shapes["U_box"]:
            L.append([box[0], box[1], box[2]])
            L.append([box[3], box[2], box[1]])
        target["L"] = shapes["L"][:]
        target["L"] += L
        target["T"] = shapes["T"][:]
        K = []
        for box in shapes["X_box"]:
            K.append([box[0], box[1], box[2], box[3]])
            K.append([box[1], box[2], box[3], box[0]])
            K.append([box[2], box[3], box[0], box[1]])
            K.append([box[3], box[0], box[1], box[2]])
        target["K"] = shapes["K"][:]
        target["K"] += K
        X = []
        for box in shapes["X_box"]:
            X.append([box[4], box[0], box[1]])
        target["X"] = shapes["X"][:]
        target["X"] += X
        target["L"] = DirectionalCornerDetectionDataset._conver_L_to_xya(target["L"])
        target["K"] = DirectionalCornerDetectionDataset._conver_K_to_xya(target["K"])
        target["X"] = DirectionalCornerDetectionDataset._conver_X_to_xya(target["X"])
        target["T"] = DirectionalCornerDetectionDataset._conver_L_to_xya(target["T"])
        return target

    @staticmethod
    def convert_valid_to_tensor(shapes: dict, img_size=[512, 512], grid_size=[16, 16], classes=["L", "T"]):
        num_classes = len(classes)
        # 修改输出张量的形状为 7×16×16
        tensor = torch.zeros([7, grid_size[0], grid_size[1]])
        for key in shapes:
            for i in range(len(shapes[key])):
                # 归一化关键点坐标
                x = shapes[key][i][0] / img_size[1]
                y = shapes[key][i][1] / img_size[0]
                angle = shapes[key][i][2]
                point = [x, y, angle]
                # 计算关键点在网格中的位置
                row, col = int(point[1] * grid_size[0]), int(point[0] * grid_size[1])
                # 计算偏移量
                offset_x = point[0] * grid_size[1] - col
                offset_y = point[1] * grid_size[0] - row
                # 计算角度的余弦和正弦值
                theta = math.radians(-point[2])
                cos_theta = math.cos(theta)
                sin_theta = math.sin(theta)
                # 填充中心坐标、cos、sin、置信度
                tensor[0, row, col] = 1  # 置信度
                tensor[1, row, col] = offset_x
                tensor[2, row, col] = offset_y
                tensor[3, row, col] = cos_theta
                tensor[4, row, col] = sin_theta
                # if key == classes[0]:
                #     tensor[5, row, col] = 1
                # else:
                #     tensor[5, row, col] = 0
                # tensor[5, row, col] = sin_theta
                # 根据类别设置类别概率
                if key in classes:
                    class_index = classes.index(key)
                    tensor[5 + class_index, row, col] = 1

        return tensor

    @staticmethod
    def convert_tensor_to_valid(tensor, img_size=[512, 512], grid_size=[16, 16], classes=["L", "T"], threshold=0.5):

        C, H, W = tensor.shape
        # 遍历每个batch
        batch_dict = {key: [] for key in classes}
        # 遍历每个位置 (h, w)
        for h in range(H):
            for w in range(W):
                # 提取当前像素的通道值
                confidence = tensor[0, h, w].item()
                if confidence > threshold:
                    x = (w + tensor[1, h, w].item()) * (img_size[1] / grid_size[1])
                    y = (h + tensor[2, h, w].item()) * (img_size[0] / grid_size[0])
                    cos = tensor[3, h, w].item()
                    sin = tensor[4, h, w].item()
                    classes_channels = tensor[5:, h, w].tolist()

                    # 计算角度（弧度）
                    angle_rad = torch.atan2(torch.tensor(sin), torch.tensor(cos)).item()
                    # 转换为角度
                    angle_deg = torch.rad2deg(torch.tensor(angle_rad)).item()

                    # 找到类别索引（取最大值的索引）
                    class_idx = classes_channels.index(max(classes_channels))
                    class_key = classes[class_idx]
                    batch_dict[class_key].append([x, y, (-angle_deg) % 360])

        return batch_dict

    @staticmethod
    def _conver_cos_and_sin_to_angle(cos, sin):
        theta1 = math.acos(cos)
        theta2 = math.asin(sin)
        angle = math.degrees(theta1)
        if theta2 < 0:
            angle = 360 - angle
        return (360 - angle) % 360

    @staticmethod
    def _conver_point_to_xya(point, point_start, point_end):
        """conver three points to xya,which first point is the starting point
        and the second and third points are the dirction

        Returns:
            _type_: _description_
        Example:
            point = [[100,120],[200,0],[0,0]]
            res = [100,120,90]
        """

        def calculate_angular_bisector(point_start, point_end):
            AB = (
                point_end[0] - point_start[0],
                point_end[1] - point_start[1],
            )
            BA_len = (AB[0] ** 2 + AB[1] ** 2) ** (1 / 2)
            cos = AB[0] / BA_len
            sin = AB[1] / BA_len
            theta1 = math.acos(cos)
            theta2 = math.asin(sin)
            angle = math.degrees(theta1)
            if theta2 < 0:
                angle = 360 - angle
            # 垂直翻转角度方向
            # 方便使用数据增强，数据增强的坐标系与图像不一致
            return -angle

        angle = calculate_angular_bisector(point_start, point_end)
        res = [point[0], point[1], angle]
        return res

    @staticmethod
    def _conver_K_to_xya(points):
        res = []
        for item in points:
            xya = DirectionalCornerDetectionDataset._conver_point_to_xya(item[1], item[1], item[3])
            res.append(xya)
        return res

    @staticmethod
    def _conver_X_to_xya(points):
        res = []
        for item in points:
            xya = DirectionalCornerDetectionDataset._conver_point_to_xya(item[0], item[1], item[2])
            res.append(xya)
        return res

    @staticmethod
    def _conver_L_to_xya(L):
        # 取L的角平分线为方向
        def calculate_angular_bisector(L: list):
            # A,B,C对应三个点
            BA = (L[0][0] - L[1][0], L[0][1] - L[1][1])
            BA_len = (BA[0] ** 2 + BA[1] ** 2) ** (1 / 2)
            cos1 = BA[0] / BA_len
            sin1 = BA[1] / BA_len
            BC = (L[2][0] - L[1][0], L[2][1] - L[1][1])
            BC_len = (BC[0] ** 2 + BC[1] ** 2) ** (1 / 2)
            cos2 = BC[0] / BC_len
            sin2 = BC[1] / BC_len
            mid_AC = ((cos1 + cos2) / 2, (sin1 + sin2) / 2)
            AC_len = (mid_AC[0] ** 2 + mid_AC[1] ** 2) ** (1 / 2)
            cos = mid_AC[0] / AC_len
            sin = mid_AC[1] / AC_len
            theta1 = math.acos(cos)
            theta2 = math.asin(sin)
            angle = math.degrees(theta1)
            if theta2 < 0:
                angle = 360 - angle
            # 垂直翻转角度方向
            # 方便使用数据增强，数据增强的坐标系与图像不一致
            return -angle + 45

        def calculate_angle(L):
            """
            计算两个点之间的角度（以弧度为单位）。

            参数:
                point1 (tuple): 第一个点的坐标，格式为 (x1, y1)
                point2 (tuple): 第二个点的坐标，格式为 (x2, y2)

            返回:
                float: 两个点之间的角度（以弧度为单位）
            """
            # 提取点的坐标
            point1, point2 = L[0], L[1]
            x1, y1 = point1
            x2, y2 = point2

            # 计算向量的坐标差
            dx = x2 - x1
            dy = y2 - y1

            # 计算角度（以弧度为单位）
            angle = math.atan2(dy, dx)
            angle_deg = math.degrees(angle)
            # 将角度控制在0到360度范围内
            angle_deg = angle_deg % 360
            return -angle_deg

        res = []
        for item in L:
            if len(item) == 3:
                angle = calculate_angular_bisector(item)
                res.append([item[1][0], item[1][1], angle])
            else:
                angle = calculate_angle(item)
                res.append([item[0][0], item[0][1], angle])

        return res

    @staticmethod
    def draw_valid_on_data(
        image,
        true_valid={},
        perdict_valid={},
        target_keypoint_type_and_colcor=[("L", (0, 255, 0)), ("T", (0, 0, 255))],
        path_img=None,
    ):
        """
        Draws the true and predicted keypoints on the image.

        Args:
            image (np.ndarray): The input image.
            true_valid (dict): A dictionary containing the true keypoints.
            perdict_valid (dict): A dictionary containing the predicted keypoints.
            target_keypoint_type_and_colcor (list): A list of tuples containing the keypoint type and color.

        Returns:
            np.ndarray: The image with the true and predicted keypoints drawn.
        """
        # Draw the legend on the image
        image = DirectionalCornerDetectionDataset._draw_legend(image, target_keypoint_type_and_colcor)
        if path_img:
            # image = DirectionalCornerDetectionDataset._draw_text_on_image(image, (Path(path_img).name))
            pass
        # Get the height and width of the image
        img_height, img_width, _ = image.shape
        # Create a copy of the image for the left side
        left_image = image.copy()
        # Draw the text "True" on the left side of the image
        left_image = DirectionalCornerDetectionDataset._draw_text_on_image(left_image, "True")

        # Iterate through the true keypoints
        for key in true_valid:
            for point in true_valid[key]:
                for v in target_keypoint_type_and_colcor:
                    # Check if the keypoint type matches
                    if key == v[0]:
                        # Get the color for the keypoint type
                        color = v[1]
                        # Draw the angle point on the left side of the image
                        left_image = DirectionalCornerDetectionDataset._draw_angle_point_on_image(
                            left_image, point, color
                        )
                        if perdict_valid:
                            # Check if the predicted keypoints for the current type are empty
                            if perdict_valid[key] == []:
                                flag = False
                            else:
                                # Check if the predicted point is close to the true point
                                flag = DirectionalCornerDetectionDataset._is_point_close(perdict_valid[key], point)
                            if not flag:
                                # Draw a red circle around the true point if the predicted point is not close
                                cv2.circle(
                                    left_image, (int(point[0]), int(point[1] )), 10, (0, 0, 255), 2
                                )
        # Create a copy of the image for the right side
        right_image = image.copy()
        # Draw the text "Predict" on the right side of the image
        right_image = DirectionalCornerDetectionDataset._draw_text_on_image(right_image, "Predict")
        # Iterate through the predicted keypoints
        for key in perdict_valid:
            # Bug fix: Should iterate through perdict_valid[key] instead of true_valid[key]
            for point in perdict_valid[key]:
                for v in target_keypoint_type_and_colcor:
                    # Check if the keypoint type matches
                    if key == v[0]:
                        # Get the color for the keypoint type
                        color = v[1]
                        # Draw the angle point on the right side of the image
                        right_image = DirectionalCornerDetectionDataset._draw_angle_point_on_image(
                            right_image, point, color
                        )
        # Check if both true and predicted keypoints exist
        if perdict_valid and true_valid:
            # Concatenate the left and right images horizontally
            res_image = cv2.hconcat([left_image, right_image])
        # Check if only true keypoints exist
        elif true_valid and not perdict_valid:
            res_image = left_image
        # Check if only predicted keypoints exist
        elif perdict_valid and not true_valid:
            res_image = right_image
        else:
            pass

        return res_image

    @staticmethod
    def draw_tensor_on_data(
        image:torch.Tensor,
        true_tensor:torch.Tensor,
        perdict_tensor:torch.Tensor,
        threshold:float=0.5,
        target_keypoint_type_and_colcor=[("L", (0, 255, 0)), ("T", (0, 0, 255))],
    ):
        img = DirectionalCornerDetectionDataset.convert_image_from_tensor_to_numpy(image)
        true_keypoint = DirectionalCornerDetectionDataset.convert_tensor_to_valid(true_tensor,threshold = threshold)
        perdict_keypoint = DirectionalCornerDetectionDataset.convert_tensor_to_valid(perdict_tensor, threshold = threshold)
        img = DirectionalCornerDetectionDataset.draw_valid_on_data(
            img, true_keypoint, perdict_keypoint, target_keypoint_type_and_colcor
        )
        return img

    @staticmethod
    def _draw_angle_point_on_image(image: np.ndarray, point: tuple, color: tuple, lenght: int = 50):
        """Draws point with direction on the image.The format of point is xya.

        Args:
            image (_type_): _description_
            point (tuple): A tuple consisting of (x,y,a). "x" is the horizontal coordinate;
                "y" is the vertical coordinate; "a" stands for angle, angle system.
            color (tuple): A tuple consisting of (B,G,R), the color of point
            lenght (int, optional): The length of the direction line. Defaults to 50.

        Returns:
            image: the image after drawing

        Example:
            point = (200,399,90)
            color = (0,255,0)
            image = cv2.imread('000.jpg')
            image = PointDataset()._draw_angle_point_on_image(image,point,color)
        """
        img_height, img_width, _ = image.shape
        if point[0] < 1 and point[1] < 1:
            start = (int(point[0] * img_width), int(point[1] * img_height))
        else:
            start = (int(point[0]), int(point[1]))
        theta = math.radians(-point[2])
        cos = math.cos(theta)
        sin = math.sin(theta)
        end = (start[0] + int(cos * lenght), start[1] + int(sin * lenght))
        cv2.arrowedLine(image, start, end, color, 2, tipLength=0.25)
        cv2.circle(image, start, 2, (100, 100, 100), 2)
        return image

    @staticmethod
    def _draw_legend(
        image,
        legend_items,
        position_ratio=(0, 0),
        font_scale=0.4,
        thickness=1,
        background_color=(255, 255, 255),
        text_color=(0, 0, 0),
    ):
        """
        在图像的右上角绘制图例。

        参数:
            image (np.ndarray): 输入图像矩阵（BGR格式）。
            legend_items (list): 图例项列表，每个项是一个元组，格式为 (label, color)。
            position_ratio (tuple): 图例的起始位置占图像宽度和高度的比例 (x_ratio, y_ratio)。
            font_scale (float): 字体大小。
            thickness (int): 字体粗细。
            background_color (tuple): 背景框的颜色。
            text_color (tuple): 文本的颜色。

        返回:
            image: 绘制后的图像矩阵。
        """
        # 获取图像的尺寸
        img_height, img_width, _ = image.shape

        # 动态计算图例的尺寸
        legend_width = int(img_width * 0.1)  # 图例宽度为图像宽度的15%
        legend_height_per_item = int(img_height * 0.03)  # 每个图例项的高度为图像高度的3%
        legend_height = len(legend_items) * legend_height_per_item + 5  # 图例总高度

        # 计算图例的起始位置
        x_start = int(img_width * position_ratio[0])
        y_start = int(img_height * position_ratio[1])

        # 创建一个白色背景的图例框
        legend_bg = np.full((legend_height, legend_width, 3), background_color, dtype=np.uint8)

        # 在图例框上绘制每个图例项
        for i, (label, color) in enumerate(legend_items):
            # 计算当前图例项的位置
            y_start_item = i * legend_height_per_item
            y_end_item = y_start_item + legend_height_per_item

            # 绘制颜色块
            cv2.rectangle(legend_bg, (5, y_start_item + 5), (20, y_start_item + 15), color, -1)
            # 绘制标签文本
            cv2.putText(
                legend_bg, label, (25, y_start_item + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness
            )

        # 将图例框绘制到图像的右上角
        image[y_start : y_start + legend_height, x_start : x_start + legend_width] = legend_bg

        return image

    @staticmethod
    def _draw_text_on_image(image, text):
        """
        在图像正上方绘制文字，并居中显示。

        参数:
            image (numpy.ndarray): 输入图像。
            text (str): 要显示的文字。

        返回:
            numpy.ndarray: 绘制文字后的图像。
        """
        # 获取图像的宽度和高度
        height, width = image.shape[:2]

        # 设置字体、颜色、大小等参数
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        font_scale = 1.0  # 字体大小
        color = (255, 255, 255)  # 文字颜色 (B, G, R)，这里设置为白色
        thickness = 2  # 文字线条的粗细

        # 获取文字的宽度和高度
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        # 计算文字的起始位置，使其居中
        x = (width - text_width) // 2  # 水平居中
        y = text_height + 10  # 垂直位置，稍微偏下一点

        # 在图像上绘制文字
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

        return image

    @staticmethod
    def _is_point_close(points_list, target_point, distance_threshold=0.01, angle_threshold=5):
        """
        判断目标点与列表中的任意一个点的距离和方向差是否小于特定阈值。

        参数:
            points_list (list): 包含点的列表，每个点的格式为 (x, y, a)。
            target_point (tuple): 目标点，格式为 (x, y, a)。
            distance_threshold (float): 归一化距离阈值。
            angle_threshold (float): 方向差阈值（以度为单位）。

        返回:
            bool: 如果目标点与列表中的任意一个点的距离和方向差都小于阈值，则返回 True，否则返回 False。
        """
        for point in points_list:
            # 提取点的坐标和方向
            x1, y1, a1 = point
            x2, y2, a2 = target_point

            # 计算两点之间的欧几里得距离
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 如果距离超过阈值，跳过该点
            if distance > distance_threshold:
                continue

            # 计算方向差（以度为单位）
            angle_diff = abs(a2 - a1)

            # 考虑方向差的周期性（0-360 度）
            angle_diff = min(angle_diff, 360 - angle_diff)

            # 判断方向差是否小于阈值
            if angle_diff < angle_threshold:
                return True

        # 如果没有找到符合条件的点，返回 False
        return False

    @staticmethod
    def draw_tensor_on_data_as_grid(images, true_tensor, predict_tensor):
        """
        Draws the true and predicted keypoints on the images in a grid.
        Args:
            images (list): A list of input images.
            true_tensor (dict): A dictionary containing the true keypoints.
            perdict_tensor (dict): A dictionary containing the predicted keypoints.
        Returns:
            np.ndarray: The images with the true and predicted keypoints drawn.
        """

        return

    @staticmethod
    def convert_ps20_to_labelme_json(path_ps20):
        """
        将PS20格式的JSON文件转换为labelme格式的JSON文件。
        参数:
            path_ps20 (str): PS20格式的JSON文件路径。
        返回:
            dict: 转换后的自定义格式的JSON数据。
        项目结构：
            -ps_json_label
            -testing
            -training
            -labelme_json
        """

        def custom_to_labelme(data, image_path):
            """
            将自定义格式转换为LabelMe格式的字典。

            参数:
                data (dict): 自定义格式的数据，包含"marks"和"slots"键。
                image_path (str): 图像文件路径。

            返回:
                dict: LabelMe格式的字典。
            """
            # 读取图像以获取尺寸信息
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("无法读取图像文件，请检查路径是否正确")
            image_height, image_width = image.shape[:2]

            # 初始化LabelMe格式的字典
            if "train" in str(image_path):
                imagePath = "../../" + image_path.parent.name + "/" + image_path.stem + ".jpg"
            else:
                imagePath = "../../../testing/" + image_path.parent.name + "/" + image_path.stem + ".jpg"
            labelme_dict = {
                "version": "5.3.0",
                "flags": {},
                "shapes": [],
                "imagePath": imagePath,
                "imageData": None,
                "imageHeight": image_height,
                "imageWidth": image_width,
            }

            # 处理marks数据
            if "marks" in data:
                if len(data["marks"]) >= 1 and not isinstance(data["marks"][0], list):
                    data["marks"] = [data["marks"]]
                for i in range(0, len(data["marks"])):
                    # 提取起点坐标
                    start_x = data["marks"][i][0]
                    start_y = data["marks"][i][1]
                    end_x = data["marks"][i][2]
                    end_y = data["marks"][i][3]
                    label_type = "T" if data["marks"][i][4] == 0 else "L"

                    # 创建LabelMe格式的shape字典
                    shape = {
                        "label": label_type,
                        "points": [[start_x, start_y], [end_x, end_y]],
                        "group_id": None,
                        "description": "",
                        "shape_type": "linestrip",
                        "flags": {},
                        "mask": None,
                    }

                    # 将shape添加到LabelMe字典的shapes列表中
                    labelme_dict["shapes"].append(shape)

            return labelme_dict

        path_ps20 = Path(path_ps20)
        dir_save_labelme = path_ps20 / "labelme_json"
        dir_save_labelme_train = dir_save_labelme / "training"
        dir_save_labelme_test = dir_save_labelme / "testing/all"
        dir_save_labelme_train.mkdir(parents=True, exist_ok=True)
        dir_save_labelme_test.mkdir(parents=True, exist_ok=True)
        dir_saves = [dir_save_labelme_train, dir_save_labelme_test]
        dir_train_image = path_ps20 / "training"
        dir_test_image = path_ps20 / "testing/all"
        dir_images = [dir_train_image, dir_test_image]
        dir_train_json = path_ps20 / "ps_json_label" / "training"
        dir_test_json = path_ps20 / "ps_json_label" / "testing" / "all"
        dir_jsons = [dir_train_json, dir_test_json]

        for dir_image, dir_json, dir_save in zip(dir_images, dir_jsons, dir_saves):
            json_paths = list(dir_json.glob("*.json"))
            for json_path in json_paths:
                with open(json_path, "r") as f:
                    data = json.load(f)
                image_path = dir_image / (json_path.stem + ".jpg")
                custom_data = custom_to_labelme(data, image_path)
                custom_json_path = dir_save / (json_path.stem + ".json")
                with open(custom_json_path, "w") as f:
                    json.dump(custom_data, f, indent=4)

        return
