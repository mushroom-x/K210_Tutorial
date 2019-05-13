# Maix 工具箱

[TOC]

## 工程说明

本工程克隆自[sipeed/Maix_Toolbox](https://github.com/sipeed/Maix_Toolbox), 并在其原有的脚本上做了删减，只保留必要的几个脚本文件，你可以在github上面下载完整版，这里只是为了给大家做个参考。

该工具箱主要用于**神经网络/深度学习模型的格式转换**。　

转换流程如下

| 步骤 | 原格式 | 目标格式 | 备注                                                     |
| ---- | ------ | -------- | -------------------------------------------------------- |
| １   | pb     | tflite   | pb是tensorflow模型文件, tflite为TFLite的模型文件         |
| ２   | tflite | kmodel   | kmodel是k210的模型文件, 将模型量化，压缩模型，加快运算。 |



工具箱的使用说明以及转换教程看下面的三篇文章：

- 3.TensorBoard模型结构可视化
- 4.TensorFlow模型转换为TFlite模型
- 5.TFLite模型转换为K210模型



## 文件结构

* `gen_pb_graph.py` 使用TensorBoard查看Tensorflow的pd格式模型文件的网络结构
* `get_nncase.sh`　下载nncase工具箱，解压至当前目录
* `images` 量化图像数据集
* `ncc`　nncase工具箱的文件夹
* `pb2tflite.sh` 将pd格式文件转换为tflite格式
* `tflite2kmodel.sh`　将tflite模型转换为kmodel格式的模型
* `workspace`　工作区，用于存放模型文件



