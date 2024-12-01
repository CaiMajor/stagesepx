#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    : trainTheModel.py
@Time    : 2024/11/28
@Author  : MajorCai
@Version : 0.1
@Desc    : #TODO
"""
from stagesepx.classifier.keras import KerasClassifier

# 训练模型文件
cl = KerasClassifier(
# 训练轮数
epochs=10
)
cl.train('C:/Users/majorcai/Documents/GitHub/stagesepx/demo/picture/training/stable_frame_fix')
cl.save_model('C:/Users/majorcai/Documents/GitHub/stagesepx/demo/model/WzDemo1.weights.h5', overwrite=True)