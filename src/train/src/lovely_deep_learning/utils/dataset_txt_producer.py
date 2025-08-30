# -*- encoding: utf-8 -*-
"""
@File    :   dataset_txt_producer.py
@Python  :   python3.8
@version :   0.0
@Time    :   2024/07/13 22:56:24
@Author  :   xiaopangdun 
@Email   :   18675381281@163.com 
@Desc    :   This is a simple example
"""
import random


class DatasetTXTProducer(object):
    @staticmethod
    def split_train_val_test(
        dataset: list, split_ratio: tuple = (0.6, 0.2, 0.2), shuffle: bool = True
    ):
        """split dataset proportionally into training set, validation set and test set

        Args:
            dataset (list): A list of tuple.
            split_ratio (tuple, optional): Ratio of split. Defaults to (0.6, 0.2, 0.2).
            shuffle (bool, optional): Shuffle list. Defaults to True.

        Returns:
            train(list): A list of tuple ,whose contain train dataset.
            val(list): A list of tuple ,whose contain val dataset.
            test(list):A list of tuple ,whose contain test dataset.
        """
        # TODA check input is 合法
        if shuffle:
            random.shuffle(dataset)
        index_train = int(len(dataset) * split_ratio[0])
        index_val = index_train + int(len(dataset) * split_ratio[1])
        # index_test = index_val + int(len(dataset) * split_ratio[2])
        train = dataset[:index_train]
        val = dataset[index_train:index_val]
        test = dataset[index_val:]
        return train, val, test

    @staticmethod
    def save_list_to_txt(contents: list, path_save: str, header: str = None):
        """
        Save a list to a text file.

        Args:
            contents (list): The list to be saved to the text file.
            path_save (str): The file path where the list will be saved.
            header (str, optional): The header string that will be written at the beginning of the file. Default is None.

        Returns:
            None

        Raises:
            IOError: If an IO error occurs.

        """
        try:
            with open(path_save, "w") as f:
                if header:
                    f.write(header)
                for line in contents:
                    temp_line = " ".join([str(item) for item in line])
                    f.write(temp_line)
                    f.write("\n")
            print("Success: Save to {}".format(path_save))
        except IOError as e:
            print("Error: Unable to save to {}. Reason: {}".format(path_save, e))


if __name__ == "__main__":

    list1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 1]
    list2 = ["a", "b", "c", "d", "e", 1, 1, 1, 1, 1, 1]

    # 打包成元组列表
    combined = list(zip(list1, list2))
    print(combined)
    producer = DatasetTXTProducer()
    train, val, test = producer.split_train_val_test(combined)

    print(train)  # 输出：(3, 2, 1, 5, 4)
    print(val)  # 输出：('c', 'b', 'a', 'e', 'd')
    print(test)  # 输出：('c', 'b', 'a', 'e', 'd')
    # # 解包成两个列表
    # list1_shuffled, list2_shuffled = zip(*combined)

    # print(list1_shuffled)  # 输出：(3, 2, 1, 5, 4)
    # print(list2_shuffled)  # 输出：('c', 'b', 'a', 'e', 'd')
    pass
