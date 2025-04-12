# -*- encoding: utf-8 -*-
"""
@File    :   labelme.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/07/13 21:41:45
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
import json
import warnings


class MyLabelme(object):
    def __init__(self, path_json: str) -> None:
        # 只能初始化已有初始化labelme标签
        # 如.json文件存在则加载
        # 否则根据路劲新建空的.json,并填充空的标签
        # 不能初始化没有.json的文件
        self.path_json = path_json
        self.version = None
        self.flags = None
        self.shapes = None
        self.imagePath = None
        self.imageData = None
        self.imageHeight = None
        self.imageWidth = None
        self.load_json()


    def load_json(self):
        # 判断路劲必须存在，且为labelme格式
        with open(self.path_json, "r") as f:
            labelme = json.load(f)
        self.version = labelme["version"]
        self.flags = labelme["flags"]
        self.shapes = labelme["shapes"]
        self.imagePath = labelme["imagePath"]
        self.imageData = labelme["imageData"]
        self.imageHeight = labelme["imageHeight"]
        self.imageWidth = labelme["imageWidth"]

    def save_json(self, path:str=None):
        labelme = {}
        labelme["version"] = self.version
        labelme["flags"] = self.flags
        labelme["shapes"] = self.shapes
        labelme["imagePath"] = self.imagePath
        labelme["imageData"] = self.imageData
        labelme["imageHeight"] = self.imageHeight
        labelme["imageWidth"] = self.imageWidth
        # 给路径则保存
        if not path:
            path = self.path_json
        with open(path, "w") as f:
            json.dump(labelme, f, indent=4)


    def set_version(self, version):
        self.version = version


    def set_flag(self, flag: str, value: bool):
        self.flags[flag] = value

    def delete_flag(self, flag):
        pass

    def set_imagePath(self, path_image):
        pass

    def set_imageHeight(self, height: int):
        pass

    def set_imageHeight(self, height: int):
        pass

    def set_imageWidth(self, width: int):
        pass


if __name__ == "__main__":
    pass
