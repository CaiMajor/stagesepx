#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    : splitTheCardingVideo.py
@Time    : 2024/11/28
@Author  : MajorCai
@Version : 0.1
@Desc    : #TODO
"""
from stagesepx.cutter import VideoCutter
from stagesepx.video import VideoObject

# 将视频切分成帧
file_name = './video/1.mp4'
video = VideoObject(file_name, pre_load=True)
# 新建帧，计算视频总共有多少帧，每帧多少ms
video.load_frames()
# 压缩视频
cutter = VideoCutter()
# 计算每一帧视频的每一个block的ssim和psnr。
res = cutter.cut(video, block=6)
# 计算出判断A帧到B帧之间是稳定还是不稳定
stable, unstable = res.get_range(threshold=0.96)
# 把分好类的稳定阶段的图片存本地
res.pick_and_save(stable, 5, to_dir='picture/training/stable_frame', meaningful_name=True)