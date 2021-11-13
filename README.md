# JX_Training

这个库制作于我的实习过程中，学习Detectron2的框架，使用了Trainer+Hook的方式在训练的过程中对因子信号模型实现监测
以及保存，让模型训练能够变得轻松惬意。现在已经实现的Hook包括TqdmHook，CustomEvalHook（挂载回测系统），BatchLossHook
（在测试组测试Loss），IterationWriterHook（记录训练过程中的某些参数/IR/Pearson/Loss，通过tensorboard查看）。


为了保护某些可能的知识产权，只放置了部分的py文件，以及模糊处理过的ipynb文件。


