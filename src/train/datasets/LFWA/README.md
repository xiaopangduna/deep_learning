# LFWA

图像只从一个链接下载。三个属性 `.mat` 在仓库里，不用再下。DataModule 还没接。

```text
https://drive.usercontent.google.com/download?id=1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA&export=download&confirm=t
```

这是 `lfw.tgz`，180566744 字节，SHA256 `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0`。13233 张 JPEG，5749 个身份，布局 `lfw/<身份>/<身份>_0001.jpg`。包内没有属性。

```bash
curl -L -o datasets/LFWA/lfw/lfw.tgz \
  "https://drive.usercontent.google.com/download?id=1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA&export=download&confirm=t"
```

仓库里的三个属性文件在 `datasets/LFWA/lfw/`：

| 文件 | 大小 | 内容 |
|---|---|---|
| `lfw_att_40.mat` | 6171224 字节 | MATLAB 7.3，2014-09-25。40 个二值属性 |
| `lfw_att_73.mat` | 6310861 字节 | MATLAB 7.3，2014-09-15。73 个属性 |
| `indices_train_test.mat` | 65227 字节 | MATLAB 7.3，2014-09-15。训练/测试划分 |

`lfwa_attr_to_csv.py` 只认文本表 `list_attr_lfwa.txt`，还不读这些 `.mat`。

## 标签

40 个二值属性在 `lfw_att_40.mat` 里。若手头已经有文本表 `list_attr_lfwa.txt`，放到数据目录后即可转 CSV：

```bash
mv ~/Downloads/list_attr_lfwa.txt datasets/LFWA/lfw/list_attr_lfwa.txt
```

原始表是空格分隔的纯文本，不适合直接当训练表：

```text
13233
5_o_Clock_Shadow Arched_Eyebrows ... Young
George_HW_Bush_0001.jpg -1 1 ... 1
```

第一行是图像数，第二行是 40 个属性名，之后每行一个扁平文件名和 40 个 `1` 或 `-1`。图实际在 `lfw/George_HW_Bush/George_HW_Bush_0001.jpg`。公开协议按文件里的顺序切：前 6263 行训练，其余 6970 行测试。没有官方验证集。

`lfwa_attr_to_csv.py` 做这三步：`-1` 改成 `0`、`1` 保持 `1`；文件名补上身份目录；按 6263 / 6970 写成两个 CSV。不打开 JPEG，图像可以还在压缩包里。

```bash
python datasets/LFWA/lfwa_attr_to_csv.py
python datasets/LFWA/lfwa_attr_to_csv.py --overwrite
```

`train.csv` 和 `test.csv` 都已存在时跳过写入，但仍检查属性文件。`--overwrite` 用于属性文件换了之后重写。

```text
path_img,5_o_Clock_Shadow,Arched_Eyebrows,...,Young
lfw/George_HW_Bush/George_HW_Bush_0001.jpg,0,1,...,1
```

`path_img` 相对 `datasets/LFWA/lfw/`。40 列的顺序与属性文件第二行一致。

`lfw_att_73.mat` 是另一套 73 个属性，取值不是 `1`/`-1`，不能代替 `lfw_att_40.mat`。`lfwa_attr_to_csv.py` 只认 40 个二值属性的文本表。

## 目录

```text
datasets/LFWA/
├── README.md
├── unpack_lfwa_plus.py
├── lfwa_attr_to_csv.py
└── lfw/
    ├── LFWA+-20261125T093552Z-1-001.zip
    ├── lfw.tgz
    ├── lfw_att_40.mat
    ├── lfw_att_73.mat
    ├── indices_train_test.mat
    ├── list_attr_lfwa.txt
    ├── train.csv
    ├── test.csv
    └── lfw/
        └── George_HW_Bush/George_HW_Bush_0001.jpg
```

## 软链接

`datasets/LFWA/` 留在项目里，放这份 README、脚本，以及 `lfw/` 下的三个 `.mat`。图像、压缩包和 CSV 不进 Git。不要把 `datasets/LFWA/lfw` 换成软链接，否则会盖住已跟踪的属性文件。图像可以放在这个目录里，Git 不会跟踪 JPEG 和压缩包。

```bash
ln -s /data/lfw/lfw datasets/LFWA/lfw/lfw
```

图像若在别的磁盘，链的是里面的 `lfw/` 图像目录，不是 `datasets/LFWA/lfw` 这一层。
