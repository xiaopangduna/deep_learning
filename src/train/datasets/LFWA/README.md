# LFWA

目前只下载并解压原始 LFW 图像。DataModule、CSV 和模型还没接。下载逻辑在 `lovely_deep_learning.data_module.lfwa.prepare_lfw_images`，本目录的 `download_lfw.py` 调用它。

## 这份包是什么

网盘上下来的 `lfw.tgz` 是原始 LFW 图像，不是带 40 个属性的 LFWA。包内没有 txt、csv 或其他标注。

| | |
|---|---|
| 分享页 | https://drive.google.com/file/d/1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA/view?usp=sharing |
| 直链 | `https://drive.usercontent.google.com/download?id=1JIgAXYqXrH-RbUvcsB3B6LXctLU9ijBA&export=download&confirm=t` |
| 大小 | 180566744 字节 |
| 图像 | 13233 张 JPEG，抽查一张为 250×250 |
| 身份 | 5749 个目录 |
| 布局 | `lfw/<身份>/<身份>_0001.jpg` |

分享页本身是 HTML 确认页，脚本请求的是上面的直链。直链返回的字节数和 `tmp/lfw.tgz` 一致。马萨诸塞大学的 [lfw.tgz](http://vis-www.cs.umass.edu/lfw/lfw.tgz) 是同一份发布；写这份说明时该地址返回 502，所以自动下载走网盘直链。

## 标签

40 个二值属性在单独的 `list_attr_lfwa.txt` 里，不在 `lfw.tgz` 中，脚本也不能自动下。[CelebA 页面](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 底部的 LFWA+ 指向 [这个网盘目录](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ3NzdzRhVUhSams)。目录要登录 Google 才能打开，里面没有单文件直链。登录后下载 `list_attr_lfwa.txt`，放到数据目录：

```bash
# 浏览器下载后
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

同一批 LFW 图有两套标注。下一步若是 CelebA，用 40 个二值属性。73 个现在就能另存，但列、取值和 CelebA 对不上。

| | 40 个：`list_attr_lfwa.txt` | 73 个：[lfw_attributes.txt](http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt) |
|---|---|---|
| 来源 | Liu 等，与 CelebA 同一套名字 | Kumar 等，哥伦比亚大学 |
| 取值 | `1` / `-1`，转成 CSV 后是 0/1 | 实数分数，可正可负，没有收成 0/1 |
| 怎么指到图 | `George_HW_Bush_0001.jpg` | `person` 加 `imagenum`，如 `Aaron Eckhart`、`1` |
| 划分 | 前 6263 行训练，其余测试 | 没有这套属性分类划分 |
| 现在能否拿到 | 要登录 Google 网盘，没有直链 | 浏览器会直接打开这段文本，另存即可，约 14.9MB |
| 多出来的 | 与 CelebA 对齐的 Young、Attractive 等 | 人种、年龄段、光照、墨镜、睁眼 |

73 个属性的文件是纯文本，浏览器会直接打开。另存为：

```bash
curl -L -o datasets/LFWA/lfw/lfw_attributes.txt \
  http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
```

40 不是 73 的子集。38 个名字能对上，`Young` 在 73 里拆成了 Baby、Child、Youth、Middle Aged、Senior，`Attractive` 拆成了 Attractive Man 和 Attractive Woman。同名列的数值也不是同一套：一边是 `1`/`-1`，一边是实数分数。不能从 73 列里抽出 40 列代替 `list_attr_lfwa.txt`。

`lfwa_attr_to_csv.py` 只认 40 个二值属性的那一份。

## 下载

在 `src/train` 下：

```bash
python datasets/LFWA/download_lfw.py
```

已经下好的包放进数据目录后再跑，就不会重新下载：

```bash
mkdir -p datasets/LFWA/lfw
mv tmp/lfw.tgz datasets/LFWA/lfw/lfw.tgz
python datasets/LFWA/download_lfw.py
```

`lfw.tgz` 大小已经是 180566744 字节时跳过下载。`lfw/` 里已经有 13233 张 JPEG 时跳过解压。解压后保留压缩包。内容张数或身份数对不上时停下，不把残包解进去。

## 目录

```text
datasets/LFWA/
├── README.md
├── download_lfw.py
├── lfwa_attr_to_csv.py
└── lfw/
    ├── lfw.tgz
    ├── list_attr_lfwa.txt
    ├── train.csv
    ├── test.csv
    └── lfw/
        └── George_HW_Bush/George_HW_Bush_0001.jpg
```

## 软链接

`datasets/LFWA/` 留在项目里，只放这份 README、`download_lfw.py` 和 `lfwa_attr_to_csv.py`。压缩包、属性文件、CSV 和图像都在 `lfw/`。数据在别的磁盘时，把整个数据目录链成 `lfw`，不要只链里面的身份目录，也不要把 `datasets/LFWA` 本身换成软链接。

```bash
ln -s /data/lfw datasets/LFWA/lfw
```

`/data/lfw` 里面就是 `lfw.tgz`、`list_attr_lfwa.txt` 和解压出来的 `lfw/`。
