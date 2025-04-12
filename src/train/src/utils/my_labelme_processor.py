# -*- encoding: utf-8 -*-
"""
@File    :   lableme_processor.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/07/14 23:04:39
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
from PIL import Image
from pathlib import Path
import json
from my_labelme import MyLabelme


class MyLabelmeProcessor(object):

    @staticmethod
    def create_new_labelme_json(dir_images: str, dir_labels: str):
        """Create an empty label file for the image.

        Make sure the image and label folder paths are at the same level.labelme version default to "5.5.0"

        Args:
            dir_images (str): Path of images.
            dir_labels (str): Path of empty floder for save labelme's json file.

        Example:
            process = MyLabelmeProcessor()
            dir_images = r"D:\A_Project\image_classification_example\dataset_sample\image"
            dir_labels = r"D:\A_Project\image_classification_example\dataset_sample\test_labels"
            process.create_new_labelme_json(dir_images,dir_labels)
        Structure:
            iamges
                1.jpg
                2.jpg
                ...
            labels
        """
        dir_images = Path(dir_images)
        dir_labels = Path(dir_labels)
        #  default label folder is the same level as the image folder
        for path in dir_images.iterdir():
            # TODO check is image
            labelme = {}
            path_relative_image = "..\\" + path.parent.name + "\\" + path.name
            temp_img = Image.open(path)
            imageWidth = temp_img.width 
            imageHeight = temp_img.height
            path_label = dir_labels.joinpath(path.stem + ".json")
            labelme["version"] = "5.5.0"
            labelme["flags"] = {}
            labelme["shapes"] = []
            labelme["imagePath"] = path_relative_image
            labelme["imageData"] = None
            labelme["imageHeight"] = imageHeight
            labelme["imageWidth"] = imageWidth
            with open(path_label, "w") as f:
                json.dump(labelme, f, indent=4)
                print(path_label)
        return 

    @staticmethod
    def set_flag_to_json(
        paths: list,
        flag:str,
        value
    ):
        # 往paths中的json文件中添加flag
        for path in paths:
            my_labelme = MyLabelme(path)
            my_labelme.set_flag(flag,value)
            my_labelme.save_json()
            print("Success: add {} to {}".format(flag,path))



if __name__ == "__main__":
    # # 根据images文件夹创建对应的json文件
    # process = MyLabelmeProcessor()
    # dir_images = r"D:\A_Project\database\bdd100k"
    # dir_labels = r"D:\A_Project\database\bdd100k_labelme"
    # process.create_new_labelme_json(dir_images, dir_labels)

    # 往指定的json文件中添加flag
    process = MyLabelmeProcessor()
    dir_labels = Path(r"D:\A_Project\database\bdd100k_labelme")
    paths = [path for path in dir_labels.iterdir()]
    process.set_flag_to_json(paths,"highway",False)

    # pass
