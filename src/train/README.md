# deep_learning

# 1.Introduce
{**以下是 Gitee 平台说明，您可以替换此简介**}


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