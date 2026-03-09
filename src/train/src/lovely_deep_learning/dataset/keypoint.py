# -*- encoding: utf-8 -*-
"""
@File    :   keypoint.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/09/12 22:03:34
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""

import json
import os
import copy
import math
import torch
from torchvision.transforms import transforms as T

import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A

from src.datasets.base_dataset import BaseDataset


class RegressionPointDataset(BaseDataset):
    """A pytorch dataset for parking slot.
    This class can read data sets in xiaopangdun format, implement
    conversions between data formats, and plot data to images.

    Args:
        Dataset (_type_): _description_


    """

    def __getitem__(self, index: int):
        """_summary_

        Args:
            index (int): _description_

        Returns:
            net_in: Image of shape [3, H, W], a pure tensor
            nei_out: a dict containing the following fields
                category, integer torch.Tensor of shape [N]: the label for each bounding box.
        """
        net_in = {}
        net_out = {}
        path_img = self.path_datas[index]
        path_label = self.path_labels[index]
        img = cv2.imread(path_img)
        with open(path_label, "r") as f:
            labelme = json.load(f)
        # get U_box,X_box,X,K,L from lableme
        shapes_raw = {
            "U_box": [],
            "X_box": [],
            "X": [],
            "K": [],
            "L": [],
        }
        shapes_raw = self.filter_shapes_from_labelme(labelme, shapes_raw)
        # conver U_box to L,conver X_box to X and K
        shapes_valid = self.convert_raw_to_valid(shapes_raw)
        if self.transforms:
            transformed = self.transforms(
                image=img,
                keypoints=shapes_valid["L"],
                keypoints1=shapes_valid["X"],
                keypoints2=shapes_valid["K"],
            )
            img = transformed["image"]
            shapes_valid["L"] = transformed["keypoints"]
            shapes_valid["X"] = transformed["keypoints1"]
            shapes_valid["K"] = transformed["keypoints2"]
        img_size = (img.shape[0], img.shape[1])  # HW
        feature_map_size = (6, 16, 16)
        shapes_tensor = self.convert_valid_to_tensor(shapes_valid, img_size, feature_map_size)
        # shapes_norm = self.conver_valid_to_norm(shapes_valid, img_size)

        # 张量化
        img_tensor = T.ToTensor()(img)

        net_in["img"] = img
        net_in["img_tensor"] = img_tensor
        net_out["keypoint_raw"] = shapes_raw
        net_out["keypoint_valid"] = shapes_valid
        net_out["keypoint_tensor"] = shapes_tensor
        return net_in, net_out

    @staticmethod
    def get_transforms_for_train():
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
                A.Resize(512, 512),
            ],
            keypoint_params=A.KeypointParams(format="xya", remove_invisible=True),
            additional_targets={
                "keypoints1": "keypoints",
                "keypoints2": "keypoints",
            },
        )
        return transforms

    @staticmethod
    def get_transforms_for_val():
        transforms = A.Compose(
            [A.Resize(512, 512)],
            keypoint_params=A.KeypointParams(format="xya", remove_invisible=True),
            additional_targets={
                "keypoints1": "keypoints",
                "keypoints2": "keypoints",
            },
        )
        return transforms

    def convert_raw_to_valid(self, shapes: dict):
        """conver U_box from dict to L, conver X_box from dict to X,K

        Args:
            shapes (dict): label in labelme.

        Returns:
            _type_: converted labels

        Example:
            shapes ={
                "U_box":[],
                "X_box":[],
                "L:[],
                "K":[],
                "X":[]
            }
            shapes_valid = self.conver_raw_to_valid(shapes)
            shapes_valid = {
                "L":[],
                "K":[],
                "X":[],
            }
        """
        res = {"L": [], "X": [], "K": []}
        L = []
        for box in shapes["U_box"]:
            L.append([box[0], box[1], box[2]])
            L.append([box[3], box[2], box[1]])
        res["L"] = shapes["L"][:]
        res["L"] += L
        K = []
        for box in shapes["X_box"]:
            K.append([box[0], box[1], box[2], box[3]])
            K.append([box[1], box[2], box[3], box[0]])
            K.append([box[2], box[3], box[0], box[1]])
            K.append([box[3], box[0], box[1], box[2]])
        res["K"] = shapes["K"][:]
        res["K"] += K
        X = []
        for box in shapes["X_box"]:
            X.append([box[4], box[0], box[1]])
        res["X"] = shapes["X"][:]
        res["X"] += X
        res["L"] = self._conver_L_to_xya(res["L"])
        res["K"] = self._conver_K_to_xya(res["K"])
        res["X"] = self._conver_X_to_xya(res["X"])
        return res

    def convert_valid_to_tensor(self, shapes: dict, img_size, feature_map_size):
        for key in shapes:
            for i in range(len(shapes[key])):
                shapes[key][i] = [
                    shapes[key][i][0] / img_size[0],
                    shapes[key][i][1] / img_size[1],
                    shapes[key][i][2],
                ]
        tensor = torch.zeros([feature_map_size[0], feature_map_size[1], feature_map_size[2]])
        for point in shapes["L"]:
            row, col = int(point[0] * feature_map_size[1]), int(point[1] * feature_map_size[1])
            template_info = [1, 1, 0, 0, 0, 0]
            template_info[2] = point[0] * feature_map_size[1] % 1
            template_info[3] = point[1] * feature_map_size[1] % 1
            theta = math.radians(-point[2])
            template_info[4] = math.cos(theta)
            template_info[5] = math.sin(theta)
            tensor[:, col, row] = torch.tensor(template_info)
        for point in shapes["X"]:
            row, col = int(point[0] * feature_map_size[1]), int(point[1] * feature_map_size[1])
            template_info = [1, 0, 0, 0, 0, 0]
            template_info[2] = point[0] * feature_map_size[1] % 1
            template_info[3] = point[1] * feature_map_size[1] % 1
            theta = math.radians(-point[2])
            template_info[4] = math.cos(theta)
            template_info[5] = math.sin(theta)
            tensor[:, col, row] = torch.tensor(template_info)
        return tensor

    @staticmethod
    def convert_tensor_to_valid(self, tensor, threshold=0.5):
        res = {"L": [], "X": [], "K": []}
        index = torch.where(tensor[0] > threshold)
        for i in range(index[0].shape[0]):
            row = index[0][i]
            col = index[1][i]
            # 将 执行度，类别，x，y，cos，sin转换为xya
            temp_info = tensor[:, row, col]
            x = (temp_info[2] + col) / 16
            y = (temp_info[3] + row) / 16
            a = self._conver_cos_and_sin_to_angle(temp_info[4], temp_info[5])
            xya = [float(x), float(y), float(a)]
            if temp_info[1] > 0.5:
                res["L"].append(xya)
            if temp_info[1] <= 0.5:
                res["X"].append(xya)
        shapes = res
        for key in shapes:
            for i in range(len(shapes[key])):
                shapes[key][i] = [
                    shapes[key][i][0],
                    shapes[key][i][1],
                    shapes[key][i][2],
                ]
        return shapes

    def convert_valid_to_raw(self):
        pass
        return

    def draw_valid_on_data(self, image, shapes):
        color_L = (255, 0, 0)
        color_X = (0, 255, 0)
        color_K = (0, 0, 255)
        for point in shapes["L"]:
            image = self._draw_angle_point_on_image(image, point, color_L)
        for point in shapes["X"]:
            image = self._draw_angle_point_on_image(image, point, color_X)
        for point in shapes["K"]:
            image = self._draw_angle_point_on_image(image, point, color_K)
        return image

    def draw_tensor_on_data(self, image: np.ndarray, tensor):
        img_size = image.shape
        shapes = self.convert_tensor_to_valid(tensor, img_size=img_size)
        image = self.draw_valid_on_data(image, shapes)
        return image

    def draw_target_and_predict_on_data(self):
        return super().draw_target_and_predict_on_data()

    def draw_target_and_predict_on_data_as_grid_for_train(self):
        return super().draw_target_and_predict_on_data_as_grid_for_train()

    @staticmethod
    def _conver_cos_and_sin_to_angle(cos, sin):
        theta1 = math.acos(cos)
        theta2 = math.asin(sin)
        angle = math.degrees(theta1)
        if theta2 < 0:
            angle = 360 - angle
        return (360 - angle) % 360

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
        start = (int(point[0] * img_width), int(point[1] * img_height))
        theta = math.radians(-point[2])
        cos = math.cos(theta)
        sin = math.sin(theta)
        end = (start[0] + int(cos * lenght), start[1] + int(sin * lenght))
        cv2.arrowedLine(image, start, end, color, 2, tipLength=0.25)
        cv2.circle(image, start, 2, (100, 100, 100), 2)
        return image

    # 转换三个点为xya
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

    def _conver_K_to_xya(self, points):
        res = []
        for item in points:
            xya = self._conver_point_to_xya(item[1], item[1], item[3])
            res.append(xya)
        return res

    def _conver_X_to_xya(self, points):
        res = []
        for item in points:
            xya = self._conver_point_to_xya(item[0], item[1], item[2])
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

        res = []
        for item in L:
            angle = calculate_angular_bisector(item)
            res.append([item[1][0], item[1][1], angle])

        return res

    @staticmethod
    def get_collate_fn_for_dataloader():
        def collate_fn(x):
            return list(zip(*x))


class HeatmapPointDataset(BaseDataset):
    """A pytorch dataset for parking slot.
    This class can read data sets in xiaopangdun format, implement
    conversions between data formats, and plot data to images.

    Args:
        Dataset (_type_): _description_


    """

    def __getitem__(self, index: int):
        """_summary_

        Args:
            index (int): _description_

        Returns:
            net_in: Image of shape [3, H, W], a pure tensor
            nei_out: a dict containing the following fields
                category, integer torch.Tensor of shape [N]: the label for each bounding box.
        """
        # 输出的格式为点和方向，heatmap
        # 从labelme中获取原始点
        # 转换为点加方向的格式
        # 关键点增强
        # 关键点点归一化
        # 关键点张量化
        net_in = {}
        net_out = {}
        path_img = os.path.join(self.root, self.imgs[index])
        path_label = os.path.join(self.root, self.labels[index])
        # Processing image data
        img = cv2.imread(path_img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 数据增强不需要管通道
        with open(path_label, "r") as f:
            labelme = json.load(f)
        shapes_raw = {"U_box": [], "L": [], "X_box": []}
        shapes_raw = self.filter_shapes_from_labelme(labelme, shapes_raw)
        shapes_valid = self.conver_raw_to_valid(shapes_raw)
        if self.transforms:
            transformed = self.transforms(
                image=img,
                keypoints=shapes_valid["L"],
                keypoints1=shapes_valid["X"],
                keypoints2=shapes_valid["K"],
            )
            img = transformed["image"]
            shapes_valid["L"] = transformed["keypoints"]
            shapes_valid["X"] = transformed["keypoints1"]
            shapes_valid["K"] = transformed["keypoints2"]
        img_size = (img.shape[0], img.shape[1])  # HW
        shapes_tensor = self.conver_valid_to_tensor(
            shapes_valid,
            img_size,
            self.cfgs["heatmap_size"],
            self.cfgs["kernel_size"],
            self.cfgs["kernel_sigma"],
            self.cfgs["threshold_min"],
            self.cfgs["threshold_max"],
        )
        img_tensor = T.ToTensor()(img)
        net_in["img"] = img
        net_in["img_tensor"] = img_tensor

        net_out["shapes_raw"] = shapes_raw
        net_out["shapes_valid"] = shapes_valid
        net_out["shapes_tensor"] = shapes_tensor

        return net_in, net_out

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def get_transforms_for_train():
        transforms = A.Compose(
            [
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Rotate(),
                A.OneOf(
                    [
                        # A.HueSaturationValue(),
                        # A.RGBShift(),
                        # A.RandomFog(),
                        A.RandomShadow(),
                        # A.RandomRain(),
                    ]
                ),
                A.Resize(512, 512),
            ],
            keypoint_params=A.KeypointParams(format="xya", remove_invisible=True),
            additional_targets={
                "keypoints1": "keypoints",
                "keypoints2": "keypoints",
            },
        )
        return transforms

    @staticmethod
    def get_transforms_for_test():
        transforms = A.Compose(
            [A.Resize(512, 512)],
            keypoint_params=A.KeypointParams(format="xya", remove_invisible=True),
            additional_targets={
                "keypoints1": "keypoints",
                "keypoints2": "keypoints",
            },
        )
        return transforms

    def conver_raw_to_valid(self, shapes: dict):
        # 将U转L
        # 将L装xya

        res = {"L": [], "X": [], "K": []}
        res["L"] = shapes["L"][:]
        L_U = self._conver_U_box_to_L(shapes["U_box"])
        res["L"] += L_U
        K = self._conver_X_box_to_K(shapes["X_box"])
        res["K"] += K
        res["X"] = shapes["X_box"][:]
        res["L"] = self._conver_L_to_xya(res["L"])
        # 提取
        return res

    def conver_valid_to_tensor(
        self,
        shapes: dict,
        image_size: tuple,
        heatmap_size: list,
        kernel_size,
        kernel_sigma,
        threshold_min,
        threshold_max,
    ):
        res = {}

        norm_L = self._conver_points_to_norm(shapes["L"], image_size)
        heatmap_L = self._conver_points_to_heatmap(
            norm_L,
            heatmap_size,
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
        res["L"] = torch.from_numpy(heatmap_L)
        res["L"] = res["L"].unsqueeze(0)
        return res

    def conver_tensor_to_valid(
        self,
        shapes: dict,
        img_size: tuple,
    ):
        res = {}
        # 转换坐标
        # 转换热力图为关键点坐标
        # 提取热力图张量
        L_heatmap = shapes["L"][0]
        # 调整WH为HW
        L_heatmap = L_heatmap.transpose(1, 0).cpu()
        # 张量转numpy
        L_heatmap = L_heatmap.detach().numpy()
        L_points = self._conver_heatmap_to_points(
            L_heatmap,
            self.cfgs["min_distance"],
            self.cfgs["threshold_abs"],
            self.cfgs["max_point"],
        )
        L_heatmap_size = L_heatmap.shape
        # HW
        # img_size
        scale = (
            img_size[0] / L_heatmap_size[0],
            img_size[1] / L_heatmap_size[1],
        )
        for point in L_points:
            point[0] *= scale[0]
            point[1] *= scale[1]
        # 转换方向
        # 先随便赋值方向
        res["L"] = self._conver_angle(L_points)
        # 缩放点至原图大小

        return res

    @staticmethod
    def _conver_heatmap_to_points(
        heatmap,
        min_distance: int,
        threshold_abs: float,
        max_point: int,
    ):
        points = peak_local_max(
            heatmap,
            min_distance=min_distance,
            threshold_abs=threshold_abs,
        )
        points = points.tolist()
        return points[:max_point]

    @staticmethod
    def _conver_angle(points):
        for point in points:
            point.append(0)
        return points

    @staticmethod
    def _conver_points_to_norm(points, image_size):
        res = []
        image_H = image_size[0]
        image_W = image_size[1]
        for point in points:
            res.append([point[0] / image_W, point[1] / image_H, point[2]])
        return res

    def _conver_points_to_heatmap(
        self,
        points: list,
        heatmap_size: tuple,
        kernel_size: int,
        kernel_sigma: float,
        threshold_min: float,
        threshold_max: float,
    ):
        kernel = self._gaussian_kernel(
            size=kernel_size,
            sigma=kernel_sigma,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
        heatmap = np.zeros(heatmap_size)
        for point in points:
            x = point[0] * heatmap_size[0]
            y = point[1] * heatmap_size[1]
            start_point = (
                int(x - kernel_size // 2),
                int(y - kernel_size // 2),
            )
            end_point = (
                start_point[0] + kernel_size,
                start_point[1] + kernel_size,
            )
            temp_kernel = kernel[
                max(-start_point[0], 0) : kernel_size - max(-(heatmap_size[0] - end_point[0]), 0),
                max(-start_point[1], 0) : kernel_size - max(-(heatmap_size[1] - end_point[1]), 0),
            ]
            heatmap[
                max(start_point[0], 0) : min(end_point[0], heatmap_size[0]),
                max(start_point[1], 0) : min(end_point[1], heatmap_size[1]),
            ] = np.where(
                heatmap[
                    max(start_point[0], 0) : min(end_point[0], heatmap_size[0]),
                    max(start_point[1], 0) : min(end_point[1], heatmap_size[1]),
                ]
                < temp_kernel,
                temp_kernel,
                heatmap[
                    max(start_point[0], 0) : min(end_point[0], heatmap_size[0]),
                    max(start_point[1], 0) : min(end_point[1], heatmap_size[1]),
                ],
            )

        return heatmap

    @staticmethod
    def _gaussian_kernel(size, sigma, threshold_min=0.0, threshold_max=1.0):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma**2)),
            (size, size),
        )
        kernel = kernel / np.max(kernel)
        kernel = np.where(kernel < threshold_min, 0, kernel)
        kernel = np.where(kernel > threshold_max, 1, kernel)
        return kernel

    @staticmethod
    def _conver_U_box_to_L(boxs: list):
        res = []
        for box in boxs:
            res.append([box[0], box[1], box[2]])
            res.append([box[3], box[2], box[1]])
        return res

    @staticmethod
    def _conver_X_box_to_K(boxs: list):
        res = []
        for box in boxs:
            res.append([box[0], box[1], box[2], box[3]])
            res.append([box[1], box[2], box[3], box[0]])
            res.append([box[2], box[3], box[0], box[1]])
            res.append([box[3], box[0], box[1], box[2]])
        return res

    @staticmethod
    def _conver_L_to_xya(L):
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
            return -angle

        res = []
        for item in L:
            angle = calculate_angular_bisector(item)
            res.append((item[1][0], item[1][1], angle))

        return res

    @staticmethod
    def _conver_X_to_xya():
        pass

    @staticmethod
    def _conver_K_to_xya():
        pass

    def draw_raw_on_image(self, image, shapes: dict):
        color_U_box = (0, 255, 0)
        color_X_box = (0, 0, 255)
        color_L = (255, 0, 0)
        for box in shapes.get("U_box", []):
            image = self._draw_points_on_image(image, box, color_U_box)
        for box in shapes.get("X_box", []):
            image = self._draw_points_on_image(image, box, color_X_box)
        for box in shapes.get("L", []):
            image = self._draw_points_on_image(image, box, color_L)
        return image

    def draw_valid_on_image(self, image, shapes: dict):
        #
        color_L = (255, 0, 0)
        # color_X = (0, 255, 0)
        # color_K = (0, 0, 255)
        for point in shapes["L"]:
            image = self._draw_angle_point_on_image(image, point, color_L)
        return image

    def draw_tensor_on_image(self, image, shapes, alpha, path):
        # 绘制热力图在图像上
        heatmap = shapes["L"][0]
        heatmap = torch.transpose(heatmap, 1, 0).cpu()
        heatmap = heatmap.detach().numpy()
        self._draw_heatmap_on_image(image, heatmap, alpha, path)

    @staticmethod
    def _draw_points_on_image(image, points, color):
        for i in range(len(points) - 1):
            start = (int(points[i][0]), int(points[i][1]))
            end = (int(points[i + 1][0]), int(points[i + 1][1]))
            cv2.arrowedLine(image, start, end, color, 2, tipLength=0.25)
        return image

    @staticmethod
    def _draw_angle_point_on_image(image, point: tuple, color: tuple, lenght: int = 50):
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
        start = (int(point[0]), int(point[1]))
        theta = math.radians(-point[2])
        cos = math.cos(theta)
        sin = math.sin(theta)
        end = (start[0] + int(cos * lenght), start[1] + int(sin * lenght))
        cv2.arrowedLine(image, start, end, color, 2, tipLength=0.25)
        cv2.circle(image, start, 2, (255, 0, 0), 2)
        return image

    @staticmethod
    def _draw_heatmap_on_image(image, heatmap, alpha, path):
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, image.shape[:2])
        plt.imshow(heatmap, cmap="hot")
        plt.imshow(image, alpha=alpha)
        plt.savefig(path)


if __name__ == "__main__":
    # test RegressionPointDataset
    path_txt = r"database_sample/parking_slot/train.txt"
    cfgs = {"threshold": 0.5}
    indexs_annotations = ("data_image", "label_keypoint")
    dataset = RegressionPointDataset(
        path_txt,
        cfgs,
        indexs_annotations,
        transforms="train",
    )
    dir_save = r"database_sample/visual"
    #
    for i in range(len(dataset)):
        net_in, net_out = dataset[i]
        img = net_in["img"]
        shapes_raw = net_out["shapes_raw"]
        shapes_valid = net_out["shapes_valid"]
        shapes_tensor = net_out["shapes_tensor"]
        img_size = (img.shape[0], img.shape[1])

        # draw_img = dataset.draw_valid_on_data(img, shapes_valid)
        # draw_img = dataset.draw_tensor_on_data(img, shapes_tensor,img_size)

        img_tensor = T.ToPILImage()(net_in["img_tensor"])
        img_tensor = np.array(img_tensor)
        draw_img = dataset.draw_tensor_on_data(img_tensor, shapes_tensor, img_size)
        path = os.path.join(dir_save, "{}.jpg".format(i))
        cv2.imwrite(path, draw_img)
        print("{}".format(dataset.path_labels[i]))

    # root = r"../../database/park_slot/"
    # dir_imgs = ["train_harbor_vital"]
    # dir_jsons = ["train_harbor_vital"]
    # # cfgs = {
    # #     "heatmap_size": [128, 128],
    # #     "kernel_size": 11,
    # #     "kernel_sigma": 1.5,
    # #     "threshold_min": 0,  # 小于该阈值的高斯核的值，置0
    # #     "threshold_max": 1.0,  # 大于该阈值的高斯核的值，置1
    # #     "threshold_abs": 0.5,  # 热力值大于阈值，即可成峰
    # #     "min_distance": 10,  # 峰与峰之间的最小距离
    # #     "max_point": 50,  # 热力图的最多峰值
    # # }
