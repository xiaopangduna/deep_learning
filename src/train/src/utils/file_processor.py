# -*- coding: utf-8 -*-
"""
@File    :   file_processor.py
@Time    :   2024/03/20 22:15:52
@Author  :   xiaopangdun
@Email  :   18675381281@163.com
@Version :   1.0
@Desc    :   None
"""
import os
import shutil
from pathlib import Path


class FileProcessor(object):

    @staticmethod
    def rename_file_order(
        dir_input: str,
        dir_output: str = None,
        initial_num: int = 1,
        prefix: str = "",
    ):
        """Copy and rename files.

        Args:
            dir_input (str): The path of floder which save original file.
            dir_output (str): The path of floder which save target file.
            initial_num (int, optional): The initial number of the file name. Defaults to 1.
            prefix (str, optional): The prefix of file name . Defaults to "".
            separator (str, optional): The separator of file name. Defaults to "_".
            suffix (str, optional): THe suffix of file name. Defaults to "".

        Example:
            path_input = r"D:/A_Project/database/park_slot/train_harbor_vital"
            path_output = r"D:/A_Project/database/park_slot/train_harbor_vital"
            initial_num = 1
            prefix = "240316"
            separator = "_"
            suffix = "03"
            processor = FileProcessor()
            processor.copy_and_rename_files(
                path_input, path_output, initial_num, prefix, separator, suffix
            )
            # file name
            # 240316_03_00001.jpg
        """
        # check dir is exits.
        if not os.path.isdir(dir_input):
            print("Error  :floder is not exit.")
            print("path of dir_input: {}".format(dir_input))
        # get file name
        names = os.listdir(dir_input)
        # names.sort(key=key)
        for name in names:
            # copy and renmae file
            path_old = os.path.join(dir_input, name)
            if dir_output and dir_input != dir_output:
                path_new = os.path.join(
                    dir_output,
                    "{}{}{}{}{:05d}{}".format(
                        prefix,
                        separator,
                        suffix,
                        separator,
                        initial_num,
                        os.path.splitext(name)[-1],
                    ),
                )
                shutil.copyfile(path_old, path_new)
                print("Success :copy {} to {}".format(path_old, path_new))
            else:
                path_new = os.path.join(
                    dir_input,
                    "{}{:05d}{}".format(
                        prefix,
                        initial_num,
                        os.path.splitext(name)[-1],
                    ),
                )
                os.rename(path_old, path_new)
                print("Success :rename {} to {}".format(path_old, path_new))
            initial_num += 1
    @staticmethod
    def rename_file(
        dir_input: str,
        dir_output: str = None,
        prefix: str = "",
    ):
        """Copy and rename files.

        Args:
            dir_input (str): The path of floder which save original file.
            dir_output (str): The path of floder which save target file.
            initial_num (int, optional): The initial number of the file name. Defaults to 1.
            prefix (str, optional): The prefix of file name . Defaults to "".
            separator (str, optional): The separator of file name. Defaults to "_".
            suffix (str, optional): THe suffix of file name. Defaults to "".

        Example:
            path_input = r"D:/A_Project/database/park_slot/train_harbor_vital"
            path_output = r"D:/A_Project/database/park_slot/train_harbor_vital"
            initial_num = 1
            prefix = "240316"
            separator = "_"
            suffix = "03"
            processor = FileProcessor()
            processor.copy_and_rename_files(
                path_input, path_output, initial_num, prefix, separator, suffix
            )
            # file name
            # 240316_03_00001.jpg
        """
        # check dir is exits.
        if not os.path.isdir(dir_input):
            print("Error  :floder is not exit.")
            print("path of dir_input: {}".format(dir_input))
        # get file name
        names = os.listdir(dir_input)
        for name in names:
            # copy and renmae file
            path_old = os.path.join(dir_input, name)
            path_new = os.path.join(
                dir_output,
                "{}{}".format(
                    prefix,
                    name,
                ),
            )
            os.rename(path_old, path_new)
            print("Success :rename {} to {}".format(path_old, path_new))

if __name__ == "__main__":

    path_input = (
        r"D:\A_Project\database\bdd100k"
    )
    path_output = (
        r"D:\A_Project\database\bdd100k"
    )
    initial_num = 1
    prefix = "BDD100k_"
    processor = FileProcessor()
    processor.rename_file_order(
        path_input, path_output, initial_num, prefix, 
    )
    # processor = FileProcessor()
    # path_aaaa = Path(r"D:\A_Project\database\temp_highway")
    # for path in path_aaaa.iterdir():
    #     path_input = path
    #     path_output = path_input
    #     prefix = "CULane_{}.MP4_".format(path_input.name[:-4])
    #     processor.rename_file(
    #         path_input, path_output,prefix
    #     )
    pass
   