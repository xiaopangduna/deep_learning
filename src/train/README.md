# deep_learning

# 1.Introduce
{**以下是 Gitee 平台说明，您可以替换此简介**}

## 常用指令

使用以下命令查看工具的完整功能和参数说明：

```bash
# 创建训练需要的数据表.csv文件
python scripts/create_dataset_table_from_multi_folders_for_train_predict.py --help
```

### Home

```bash
python scripts/create_dataset_table_from_multi_folders_for_train_predict.py \
  --headers "data_img,label_detect_yolo" \
  --dirs "datasets/coco8/images/train" \
  --dirs "datasets/coco8/labels/train" \
  --suffix-groups ".jpg,.png" \
  --suffix-groups ".txt" \
  --output-dir "datasets/coco8" \
  --split-ratio "1.0" \
  --output-names "train.csv" \
  --relative-to "datasets/coco8" \
  --allow-missing \
  --shuffle \
  --verbose 
```

#### 软件架构
软件架构说明


#### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

#### 使用说明

1.  xxxx
2.  xxxx
3.  xxxx

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)

# 最终目标
将所有与深度学习相关的模型训练全部整合至一起
具体的实现方式
所有的模型训练，断点训练，onnx等导出，剪枝，量化，统一由module管理，scripts调用module
src：提供所有的源码
    核心功能：提供，datasets，losser，metrics，models，modules（负责对前面的调用和组合）
scripts：负责对源码进行调用
    核心功能：训练模型、模型推理，模型测试，剪枝训练，量化训练，制作训练集
test：测试源码
cmd.txt:所有训练指令
使用lightning cli
# 一个config文件包括训练，重训练，剪枝，量化，

yolo-detection
下载coco8数据集-

# 示例
 NMIST
 
python scripts/train.py fit --config configs/experiments/mnist.yaml


# 基础用法

所有 configs/experiments 目录下的配置文件都应支持以下指令。

bash ./scripts/run_batch_train.sh 

## 1 训练

```bash
python scripts/train.py fit --config configs/experiments/image_classifiter.yaml
```

## 2 快速测试训练流程

用于快速验证数据加载、模型结构和训练流程是否正常。

```bash
python scripts/train.py fit \
    --config configs/experiments/image_classifiter.yaml \
    --trainer.fast_dev_run true
```

## 3 断点训练（Resume Training）

指定 version 可以保证继续在同一日志文件夹下训练，否则会新建新的 version。

```bash
python scripts/train.py fit \
    --config logs/image_classifier/version_8/config.yaml \
    --ckpt_path logs/image_classifier/version_8/checkpoints/epoch=1-step=18938.ckpt \
    --trainer.logger.init_args.version 8
```

## 4 验证（Validation）

```bash
python scripts/train.py validate \
    --config logs/image_classifier/version_0/config.yaml \
    --ckpt_path logs/image_classifier/version_0/checkpoints/epoch\=25-step\=15392.ckpt
```

## 5 测试（Test）

```bash
python scripts/train.py test \
    --config logs/image_classifier/version_0/config.yaml \
    --ckpt_path logs/image_classifier/version_0/checkpoints/epoch\=25-step\=15392.ckpt
```

## 6 预测（Predict）

```bash
python scripts/train.py predict \
    --config logs/image_classifier/version_0/config.yaml \
    --ckpt_path logs/image_classifier/version_0/checkpoints/epoch\=25-step\=15392.ckpt
```

# 训练过程查看

使用 TensorBoard 查看训练日志：

```bash
tensorboard --logdir logs
```

# 图像分类

## 1 自定义数据集

数据集的读取全部依赖 CSV 文件。

用于训练的 CSV 必须包含以下列：

```
path_img, class_name, class_id
```

用于预测的 CSV 只需要：

```
path_img
```

类别名称可以通过修改以下属性进行映射：

```
lovely_deep_learning.data_module.image_classifier.ImageClassifierDataModule.map_class_id_to_class_name
```

也可以直接修改配置文件 yaml。

## 2 生成 CSV 文件

### Step 1 生成图片路径 CSV

```bash
python scripts/create_csv_to_save_path.py \
    datasets/IMAGENETTE/imagenette2-320/train \
    --relative-to datasets/IMAGENETTE \
    --header path_img \
    -o train.csv
```

### Step 2 从路径中提取类别名称

```bash
python scripts/add_column_from_path.py \
    train.csv \
    --path-col path_img \
    --pos -2 \
    --col-name class_name
```

### Step 3 添加 class_id

```bash
python scripts/add_class_id.py train.csv datasets/IMAGENETTE/info.yaml
```

### 训练示例

```bash
python scripts/train.py fit --config configs/experiments/image_classifiter.yaml
```

# 公开数据集

MNIST
```bash
python scripts/train.py fit   --config configs/experiments/image_classifiter_MNIST.yaml
```

## Imagenette
训练和验证ACC都是99%
```bash
python scripts/train.py fit   --config configs/experiments/image_classifiter_IMAGE_NETTE.yaml
```


## ImageNet
查看大型csv中的内容
cd datasets/IMAGENET
head -n 10 train.csv
# 复制少量文件验证
mkdir -p tmp/preview && tail -n +2 train.csv | awk -v k=5 'BEGIN{srand()} NR<=k{pool[NR]=$0;next} {j=int(rand()*NR)+1;if(j<=k)pool[j]=$0} END{for(i=1;i<=k;i++)print pool[i]}' | while IFS= read -r line; do echo "$line"; cp -- "${line%%,*}" tmp/preview/; done


pytest -m cli
pytest -m "not cli"