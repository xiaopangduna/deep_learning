from .base_dataset import BaseDataset


class YoloDataset(BaseDataset):
    """A basic for building pytorch model input and output tensor

    This class inherits the Dataset(from torch.utils.data) to ensure the way of load dataset ,
    visualize data and label and data enhancemment is same.

    Args:
        path_txt (str): The path of txt file ,whose contents the paths of data and label.
        paths_data (list[str]): A list of paths of data.
        paths_label (list[str]):A list of paths of label.
        transfroms (str): One of train,val,test and  none.Default
    """

    def __init__(
        self,
        path_txt,
        paths_data,
        paths_label,
        transforms=None,
    ):
        pass

    def __getitem__(self, index):
        net_in, net_out = {}, {}
        path_data = self.paths_data[index]

        path_label = self.paths_label[index]
        if path_label:
            try:
                with open(path_label, "r") as f:
                    lines = f.readlines()
                boxes = []
                for line in lines:
                    box = [float(x) for x in line.strip().split()]
                    boxes.append(box)
                boxes = np.array(boxes)  # shape: (num_boxes, 5) [class, x_center, y_center, width, height]
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                net_out["boxes"] = boxes_tensor
            except:
                warnings.warn("{} is not a valid label".format(path_label))
        data = cv2.imread(path_data)
        if self.transforms:
            transformed = self.transforms(image=data)
            data = transformed["image"]
        data_tensor = T.ToTensor()(data)
        net_in["img"] = data
        net_in["img_tensor"] = data_tensor

        return net_in, net_out
