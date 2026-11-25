# LFWA

DataModule 还没接。下载有两条途径，先走第一条。

## 首要途径：下载整个文件夹

打开 [LFWA+ 网盘目录](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams?resourcekey=0-Kpdd6Vctf-AdJYfS55VULA)，下载整个文件夹，不要逐个点文件。入口也列在 [CelebA 页面](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 底部的 LFWA+。带 `resourcekey` 才能打开。

浏览器得到 `LFWA+-<时间>Z-<卷数>-<序号>.zip`。这次只有一卷，当前这份是 `datasets/LFWA/lfw/LFWA+-20261125T093552Z-1-001.zip`，185 MB。时间每次下载都会变。放到 `datasets/LFWA/lfw/` 后解开：

```bash
unzip ~/Downloads/LFWA+-*.zip -d /tmp/lfwa-plus
mv /tmp/lfwa-plus/LFWA+/*.mat datasets/LFWA/lfw/
```

解压后是 `LFWA+/` 里的三个 `.mat` 和 `lfw.zip`。三个 `.mat` 移到 `datasets/LFWA/lfw/` 即可。仓库里已经有这三个文件，字节数一致时不必覆盖。图像已经解开时不必再动 `lfw.zip`。还没有图像时，再把它解到同一层，里面的 `lfw/` 会落到 `datasets/LFWA/lfw/lfw/`：

```bash
unzip /tmp/lfwa-plus/LFWA+/lfw.zip -d datasets/LFWA/lfw
```

2026-11-25 打开这份 zip，四个成员都在 `LFWA+/` 下：

| 成员 | 大小 | 内容 |
|---|---|---|
| `lfw.zip` | 191778528 字节 | 13233 张 JPEG，5749 个身份，`lfw/<身份>/<身份>_0001.jpg` |
| `lfw_att_40.mat` | 6171224 字节 | MATLAB 7.3，2014-09-25。40 个二值属性 |
| `lfw_att_73.mat` | 6310861 字节 | MATLAB 7.3，2014-09-15。73 个属性 |
| `indices_train_test.mat` | 65227 字节 | MATLAB 7.3，2014-09-15。训练/测试划分 |

三个 `.mat` 的位置：

```text
datasets/LFWA/lfw/lfw_att_40.mat
datasets/LFWA/lfw/lfw_att_73.mat
datasets/LFWA/lfw/indices_train_test.mat
```

## 次要途径：只下载图像

整包下不下来时，只下图像，再把三个 `.mat` 放到 `datasets/LFWA/lfw/`。这三个文件已经在仓库里，从仓库拷过去即可；若是从别处下到的，移到同一位置。

图像只有这一条链接，得到 `lfw.tgz`，180566744 字节，SHA256 `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0`。13233 张 JPEG，5749 个身份。包内没有属性。`lfw.zip` 和这份 `lfw.tgz` 是同一批图：路径一一对应，字节数和 SHA256 都相同。

```text
https://drive.usercontent.google.com/download?id=1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA&export=download&confirm=t
```

```bash
curl -L -o datasets/LFWA/lfw/lfw.tgz \
  "https://drive.usercontent.google.com/download?id=1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA&export=download&confirm=t"
```

三个 `.mat` 的目标路径：

```text
datasets/LFWA/lfw/lfw_att_40.mat
datasets/LFWA/lfw/lfw_att_73.mat
datasets/LFWA/lfw/indices_train_test.mat
```

## 目录

```text
datasets/LFWA/
├── README.md
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

`datasets/LFWA/` 留在项目里，放这份 README 和脚本。三个 `.mat` 进 Git。图像、压缩包和 CSV 不进 Git。

## 软链接

不要把 `datasets/LFWA/lfw` 换成软链接，否则会盖住已跟踪的三个 `.mat`。图像若在别的磁盘，链里面的图像目录：

```bash
ln -s /data/lfw/lfw datasets/LFWA/lfw/lfw
```

## 标签

40 个二值属性在 `lfw_att_40.mat`。`lfw_att_73.mat` 是另一套 73 个属性，取值不是 `1`/`-1`，不能代替 40 个属性那一份。`indices_train_test.mat` 是训练/测试划分。

`lfwa_attr_to_csv.py` 还不读 `.mat`。它只认文本表 `list_attr_lfwa.txt`。若手头已经有这份文本表，放到数据目录后转 CSV：

```bash
mv ~/Downloads/list_attr_lfwa.txt datasets/LFWA/lfw/list_attr_lfwa.txt
python datasets/LFWA/lfwa_attr_to_csv.py
python datasets/LFWA/lfwa_attr_to_csv.py --overwrite
```

原始表是空格分隔的纯文本：

```text
13233
5_o_Clock_Shadow Arched_Eyebrows ... Young
George_HW_Bush_0001.jpg -1 1 ... 1
```

第一行是图像数，第二行是 40 个属性名，之后每行一个扁平文件名和 40 个 `1` 或 `-1`。图实际在 `lfw/George_HW_Bush/George_HW_Bush_0001.jpg`。公开协议按文件里的顺序切：前 6263 行训练，其余 6970 行测试。没有官方验证集。

脚本做这三步：`-1` 改成 `0`、`1` 保持 `1`；文件名补上身份目录；按 6263 / 6970 写成两个 CSV。不打开 JPEG，图像可以还在压缩包里。`train.csv` 和 `test.csv` 都已存在时跳过写入，但仍检查属性文件。`--overwrite` 用于属性文件换了之后重写。

```text
path_img,5_o_Clock_Shadow,Arched_Eyebrows,...,Young
lfw/George_HW_Bush/George_HW_Bush_0001.jpg,0,1,...,1
```

`path_img` 相对 `datasets/LFWA/lfw/`。40 列的顺序与属性文件第二行一致。
