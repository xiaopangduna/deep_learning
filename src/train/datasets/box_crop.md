# 框级分类

布局以 `lovely_deep_learning.data_module.image_classifier.ImageClassifierFromDetectBoxDataModule` 为准。

## 类与配置

- 类：`lovely_deep_learning.data_module.image_classifier.ImageClassifierFromDetectBoxDataModule`
- 这个类没有 `dataset_dir`，也没有 `prepare_data` 下载或写表。训练前先准备好 CSV。

用到它的实验：

| 配置 | CSV |
|------|-----|
| `image_classifiter_from_detect_box_head_eye_open_v000.yaml` 至 `v002.yaml`、`v100.yaml`，以及 `..._mobilenet_v2_v000.yaml` | `datasets/head_eye_open/train_v301.csv`、`test_v301.csv` |
| `..._mobilenet_v2_v001.yaml` | `/home/huangwenhua/project/dataset/eye/v301_eye/` |
| `..._mobilenet_v2_v002.yaml`、`..._v002_1.yaml` | `/home/huangwenhua/project/dataset/eye/v302_eye/` |
| `box_crop_person_car_COCO.yaml` | `datasets/COCO_class/{train,val,test,predict}.csv` |
| `box_crop_person_car_COCO_age.yaml` | `datasets/head_age/train_v012.csv`、`test_v002.csv` |

CSV 可以放在 `datasets/` 下，也可以写绝对路径。`path_img` 相对 **该 CSV 所在目录** 解析。

## 怎么得到数据

原图保持 YOLO 目录，不复制。`scripts/yolo_dir_to_box_crop_csv.py` 按框展开，不筛选类别，不改 `class_id`。数据根目录需要：

```text
<data-root>/
├── dataset.yaml          # names：id 字典或名字列表
├── images/<split>/*
├── labels/<split>/*.txt
└── <split>.csv           # 脚本写出
```

`images` 与 `labels` 下同名的子目录各写成一张 `<data-root>/<split>.csv`。只在一侧出现的划分、或同一划分里 stem 对不上，会直接报错。空标签（负样本图）不写行。图像若是符号链接，先解析到真实文件再算相对路径。

```bash
python scripts/yolo_dir_to_box_crop_csv.py \
  --data-root datasets/head_eye_open
```

划分目录叫 `train_v301` 时，产物就是 `train_v301.csv`。人和车那组实验把表放在 `datasets/COCO_class/`，眼睛和年龄的表放在各自的数据根目录；仓库外的眼睛数据同样用这个脚本，再把 YAML 里的路径指过去。

## CSV

带标签的表：

```text
path_img,class_name,class_id,cx,cy,w,h,img_w,img_h
images/train_v301/0001.jpg,eye_open,0,0.51,0.42,0.08,0.04,1920,1080
```

`cx cy w h` 与 YOLO txt 相同，相对宽高归一化。`img_w` / `img_h` 是原图像素尺寸。预测表可以不带 `class_name` 和 `class_id`，其余列仍要在，因为裁剪要用框。

训练哪些 id 由 YAML 的 `class_groups` 决定：`{训练 id: [原始 class_id, ...]}`。未列出的原始类丢掉。值为 `background` 的组没有 CSV 行，只在训练集上按 `background_ratio` 采样；验证、测试、预测不加背景。短边过滤用 `min_side_px`，裁剪放大用 `box_scale`，都在 Dataset 里做，脚本不筛。

## YAML

```yaml
data:
  class_path: lovely_deep_learning.data_module.image_classifier.ImageClassifierFromDetectBoxDataModule
  init_args:
    box_scale: 1.2
    min_side_px: 16
    train_csv_paths: [datasets/head_eye_open/train_v301.csv]
    val_csv_paths: [datasets/head_eye_open/test_v301.csv]
    test_csv_paths: [datasets/head_eye_open/test_v301.csv]
    predict_csv_paths: [datasets/head_eye_open/test_v301.csv]
    key_map:
      img_path: path_img
      class_name: class_name
      class_id: class_id
      cx: cx
      cy: cy
      w: w
      h: h
      img_w: img_w
      img_h: img_h
    predict_key_map:
      img_path: path_img
      cx: cx
      cy: cy
      w: w
      h: h
      img_w: img_w
      img_h: img_h
    class_groups:
      0: [0]
      1: [2]
```
