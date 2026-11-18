# datasets

图片、压缩包和 CSV 不进 Git。每份数据的下载和目录写在该目录的 `README.md` 里，并与 `src/lovely_deep_learning/data_module` 里的类对齐。

| 目录 | 类 | 说明 |
|------|----|------|
| `MNIST/` | `MNISTDataModule` | [README.md](MNIST/README.md) |
| `IMAGENETTE/` | `ImageNetteDataModule` | [README.md](IMAGENETTE/README.md) |
| `IMAGENET/` | `ImageNetDataModule` | [README.md](IMAGENET/README.md) |
| `COCO8/` | `COCO8DataModule` | [README.md](COCO8/README.md) |
| `COCO/` | `COCODataModule` | [README.md](COCO/README.md) |
| `head_eye_open/`、`head_age/`、`COCO_class/`，或仓库外的绝对路径 | `ImageClassifierFromDetectBoxDataModule` | [box_crop.md](box_crop.md) |

在 `src/train` 下首次 `fit` 时，MNIST、ImageNette、COCO8、COCO 会由对应 DataModule 准备文件并写 CSV。ImageNet 把压缩包和解压结果放在 `datasets/IMAGENET/imagenet`（可以是软链接），再用 `datasets/IMAGENET/imagenet_to_csv.py` 或首次 `fit` 写 CSV。框级分类要事先放好 CSV。
