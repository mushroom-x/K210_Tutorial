# maixpy-tic-tac-toe
用MaixPy+机械臂完成井字棋游戏



![tic-tac-toe.jpg](./image/tic-tac-toe.jpg)



## 教程列表

[1.手写字母数据预处理](1.手写字母XO数据预处理/手写字母XO数据预处理.md)

先尝试做一下分类，目标是通过神经网络分类，然后完成Tic-Tac-Toe的游戏．

数据集可以来自于手写字母，提取字母中的`X`跟`O` 

[2.Tensorflow神经网络模型训练与冻结](2.Tensorflow神经网络模型训练与冻结/Tensorflow神经网络模型训练与冻结.md)

使用Tensorflow构建一个神经网络模型，用于识别字符`X`跟字符`O`.  使用上节课处理过的训练数据, 对模型进行训练. 训练完成之后，冻结模型, 导出pb模型文件。

[3.模型结构可视化TensorBoard](3.模型结构可视化TensorBoard/模型结构可视化TensorBoard.md)

通过TensorBoard查看神经网络模型结构

[4.TensorFlow模型转换为TFlite模型](4.TensorFlow模型转换为TFlite模型/TensorFlow模型转换为TFlite模型.md)

Tensorflow模型的后缀是`pb`, TFLite模型的后缀是`tflite`. 

从`pb`格式转换为`tflite`格式, 需要使用`Maix_Toolbox` 根目录下的`pb2tflite.sh`脚本。

[5.TFLite模型转换为K210模型](5.TFLite模型转换为K210模型/TFLite模型转换为K210模型.md)

使用nncase工具箱与MaixToolbox里面的工具完成K210模型转换。

[6.配置K210的开发环境](6.配置K210的开发环境/配置K210的开发环境.md)

