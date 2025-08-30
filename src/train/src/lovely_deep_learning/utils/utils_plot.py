import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

class UtilsPlot:
    @staticmethod
    def generate_titles(image_paths, titles=None):
        """
        根据 image_paths 和 titles 生成最终的 titles 列表

        参数:
            image_paths (list): 包含图片路径（字符串）或图像数据（np.ndarray）的列表。
            titles (list): 包含要绘制在图像左上角的字符串的列表，默认为 None。

        返回:
            list: 生成的 titles 列表。
        """
        num_images = len(image_paths)
        if titles is None:
            titles = []  # 初始化 titles 为空列表
            for idx, image_data in enumerate(image_paths):
                if isinstance(image_data, str):
                    # 如果是图片路径，提取文件名
                    titles.append(os.path.basename(image_data))
                elif isinstance(image_data, np.ndarray):
                    # 如果是 np.ndarray，使用序号
                    titles.append(f"Image {idx + 1}")
                else:
                    raise TypeError("image_paths 中的元素必须是字符串（路径）或 np.ndarray（图像数据）")
        else:
            # 如果 titles 的长度小于图片数量，按顺序补充缺失的部分
            if len(titles) < num_images:
                for idx in range(len(titles), num_images):
                    if isinstance(image_paths[idx], str):
                        # 如果是图片路径，提取文件名
                        titles.append(os.path.basename(image_paths[idx]))
                    elif isinstance(image_paths[idx], np.ndarray):
                        # 如果是 np.ndarray，使用序号
                        titles.append(f"Image {idx + 1}")
                    else:
                        raise TypeError("image_paths 中的元素必须是字符串（路径）或 np.ndarray（图像数据）")
        return titles

    @staticmethod
    def calculate_subplot_layout(num_images, nrows=None, ncols=None):
        """
        计算子图的行数和列数

        参数:
            num_images (int): 图片的数量。
            nrows (int): 子图的行数，默认为 None。
            ncols (int): 子图的列数，默认为 None。

        返回:
            tuple: (nrows, ncols) 子图的行数和列数。
        """
        if nrows is None and ncols is None:
            # 自动计算 nrows 和 ncols
            ncols = math.ceil(math.sqrt(num_images))  # 列数为平方根向上取整
            nrows = math.ceil(num_images / ncols)     # 行数为总数除以列数向上取整
        elif nrows is None:
            # 仅指定 ncols，自动计算 nrows
            nrows = math.ceil(num_images / ncols)
        elif ncols is None:
            # 仅指定 nrows，自动计算 ncols
            ncols = math.ceil(num_images / nrows)
        return nrows, ncols

    @staticmethod
    def plot_images(axes, image_paths, titles, nrows, ncols):
        """
        在子图中绘制图片并添加标题

        参数:
            axes (np.ndarray): 子图的坐标轴数组。
            image_paths (list): 包含图片路径（字符串）或图像数据（np.ndarray）的列表。
            titles (list): 包含要绘制在图像左上角的字符串的列表。
            nrows (int): 子图的行数。
            ncols (int): 子图的列数。
        """
        num_images = len(image_paths)
        for idx in range(nrows * ncols):
            row = idx // ncols  # 计算当前子图的行索引
            col = idx % ncols   # 计算当前子图的列索引
            ax = axes[row, col]

            if idx < num_images:
                # 获取当前图像数据或路径
                image_data = image_paths[idx]

                # 判断输入类型
                if isinstance(image_data, str):
                    # 如果是字符串，读取图片
                    image = cv2.imread(image_data)
                    if image is None:
                        raise ValueError(f"无法读取图片: {image_data}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB
                elif isinstance(image_data, np.ndarray):
                    # 如果是 np.ndarray，直接使用
                    image = image_data
                else:
                    raise TypeError("image_paths 中的元素必须是字符串（路径）或 np.ndarray（图像数据）")

                # 显示图片
                ax.imshow(image)

                # 在图像左上角绘制 titles 中的字符串
                ax.text(
                    10,  # x 坐标（距离左侧 10 像素）
                    20,  # y 坐标（距离顶部 20 像素）
                    titles[idx],  # 要绘制的文本
                    color='white',  # 文本颜色
                    fontsize=12,    # 字体大小
                    backgroundcolor='black'  # 背景颜色
                )
            else:
                # 显示空白图
                ax.axis('off')  # 关闭坐标轴

            ax.axis('off')  # 关闭坐标轴

    @staticmethod
    def save_images(image_paths, titles, dir_save):
        """
        保存图片到指定文件夹

        参数:
            image_paths (list): 包含图片路径（字符串）或图像数据（np.ndarray）的列表。
            titles (list): 包含要绘制在图像左上角的字符串的列表。
            dir_save (str): 保存图片的文件夹路径。
        """
        for idx, image_data in enumerate(image_paths):
            # 获取当前图像数据或路径
            if isinstance(image_data, str):
                # 如果是字符串，读取图片
                image = cv2.imread(image_data)
                if image is None:
                    raise ValueError(f"无法读取图片: {image_data}")
            elif isinstance(image_data, np.ndarray):
                # 如果是 np.ndarray，直接使用
                image = image_data
            else:
                raise TypeError("image_paths 中的元素必须是字符串（路径）或 np.ndarray（图像数据）")

            # 在图像上绘制 titles 中的字符串
            image_with_text = image.copy()
            cv2.putText(
                image_with_text,
                titles[idx],
                (10, 30),  # 文本位置（距离左侧 10 像素，距离顶部 30 像素）
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
                1,  # 字体大小
                (255, 255, 255),  # 文本颜色（白色）
                2,  # 文本厚度
                cv2.LINE_AA  # 抗锯齿
            )

            # 保存图片
            save_path = os.path.join(dir_save, f"image_{idx + 1}.png")
            cv2.imwrite(save_path, cv2.cvtColor(image_with_text, cv2.COLOR_RGB2BGR))
            print(f"图片已保存至: {save_path}")

    @staticmethod
    def plot_image_subplots_flexible(image_paths, titles=None, nrows=None, ncols=None, figsize=(10, 8), dir_save=None):
        """
        灵活绘制子图布局并显示图片，将 titles 中的字符串绘制在图像左上角，并支持保存图片

        参数:
            image_paths (list): 包含图片路径（字符串）或图像数据（np.ndarray）的列表。
            titles (list): 包含要绘制在图像左上角的字符串的列表，默认为 None。
            nrows (int): 子图的行数，默认为 None。
            ncols (int): 子图的列数，默认为 None。
            figsize (tuple): 整个画布的大小，默认为 (10, 8)。
            dir_save (str): 保存图片的文件夹路径，默认为 None（不保存）。
        """
        # 检查输入数据的长度
        num_images = len(image_paths)

        # 生成 titles
        titles = UtilsPlot.generate_titles(image_paths, titles)

        # 如果指定了保存路径，检查文件夹是否存在，如果不存在则创建
        if dir_save is not None:
            os.makedirs(dir_save, exist_ok=True)

        # 计算子图的行数和列数
        nrows, ncols = UtilsPlot.calculate_subplot_layout(num_images, nrows, ncols)

        # 创建子图
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # 如果只有一个子图，将 axes 转换为二维数组以便统一处理
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)

        # 绘制图片
        UtilsPlot.plot_images(axes, image_paths, titles, nrows, ncols)

        # 调整子图之间的间距
        plt.tight_layout()

        # 显示图形
        plt.show()

        # 如果指定了保存路径，保存图片
        if dir_save is not None:
            UtilsPlot.save_images(image_paths, titles, dir_save)