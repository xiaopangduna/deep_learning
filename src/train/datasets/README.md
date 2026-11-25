# datasets

图片、压缩包和 CSV 不进 Git。LFWA 的三个属性 `.mat` 是例外。每份数据的下载和目录写在该目录的 `README.md` 里，并与 `src/lovely_deep_learning/data_module` 里的类对齐。

| 目录 | 类 | 说明 |
|------|----|------|
| `MNIST/` | `MNISTDataModule` | [README.md](MNIST/README.md) |
| `IMAGENETTE/` | `ImageNetteDataModule` | [README.md](IMAGENETTE/README.md) |
| `IMAGENET/` | `ImageNetDataModule` | [README.md](IMAGENET/README.md) |
| `COCO8/` | `COCO8DataModule` | [README.md](COCO8/README.md) |
| `COCO/` | `COCODataModule` | [README.md](COCO/README.md) |
| `head_eye_open/`、`head_age/`、`COCO_class/`，或仓库外的绝对路径 | `ImageClassifierFromDetectBoxDataModule` | [box_crop.md](box_crop.md) |
| `LFWA/` | 尚未接 DataModule | 先下网盘整包。下不了时只下图像，三个 `.mat` 放到 `lfw/`。见 [README.md](LFWA/README.md) |

在 `src/train` 下首次 `fit` 时，MNIST、ImageNette、COCO8 会由对应 DataModule 准备文件并写 CSV。ImageNet 把压缩包和解压结果放在 `datasets/IMAGENET/imagenet`（可以是软链接），再用 `datasets/IMAGENET/imagenet_to_csv.py` 或首次 `fit` 写 CSV。COCO 把压缩包和解压结果放在 `datasets/COCO/coco`（可以是软链接）；首次 `fit` 或 `validate` 会下载缺失的标签、train2017、val2017、test2017 并解压，再写 CSV。已经手动解压时，用 `datasets/COCO/coco_to_csv.py` 只写 CSV。框级分类要事先放好 CSV。LFWA 先从网盘下载整个文件夹，解开后把三个 `.mat` 移到 `datasets/LFWA/lfw/`。整包下不了时只下 `lfw.tgz`，`.mat` 仍放到同一位置。三个 `.mat` 就位后，用 `datasets/LFWA/lfwa_attr_to_csv.py` 写成 40 属性和 73 属性各一套 CSV。DataModule 还没接。
