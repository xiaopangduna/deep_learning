    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = deepcopy(self.samples[index])

        img, img_shape = read_img(sample["img_path"],sample["img_npy_path"])
        img_tensor = self.convert_img_from_numpy_to_tensor(img)
        img_tv = tv_tensors.Image(img_tensor)
        
        bboxes_np = sample['bboxes']  # 归一化xywh，形状(N,4)
        H, W = img_shape[0], img_shape[1]  # 原始图像高宽
        bboxes_abs_xyxy = self.convert_bboxes_from_relative_to_absolute(bboxes_np, (H, W))
        bboxes_tensor = torch.from_numpy(bboxes_abs_xyxy)
        