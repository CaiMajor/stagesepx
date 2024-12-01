#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    : predictOtherVideos.py
@Time    : 2024/11/28
@Author  : MajorCai
@Version : 0.1
@Desc    : #TODO
"""
import sys
import time

from stagesepx.cutter import VideoCutter
from stagesepx.reporter import Reporter
from stagesepx.video import VideoObject
from stagesepx.classifier.keras import KerasClassifier
from loguru import logger
import result_data


def process_video_with_params(file_name, param_list, cl, cutter):
    """
    使用不同参数处理视频并进行分类
    
    Args:
        file_name (str): 视频文件路径
        param_list (list): 参数列表，每个元素为 [compress_rate, threshold, offset, block, date]
        cl: 分类器实例
        cutter: 视频切割器实例
    """
    # 获取视频名去掉后缀
    video_name = file_name.split('/')[-1].split('.')[0]
    video = VideoObject(file_name)
    video.load_frames()

    # 遍历参数列表
    for params in param_list:
        compress_rate, threshold, offset, block, date = params
        logger.info(f"当前参数: 压缩率={compress_rate}, 门限值={threshold}, 偏移量={offset}, block={block}")

        # 计算每一帧视频的每一个block的ssim和psnr
        res = cutter.cut(video, block=block)

        # 判断A帧到B帧之间是稳定还是不稳定
        stable, unstable = res.get_range(threshold=threshold, offset=offset)

        # 生成唯一的输出目录名
        output_dir = f'./picture/forecast_{video_name}_{threshold}_{offset}_{block}'

        # 把分好类的稳定阶段的图片存本地
        res.pick_and_save(stable, 20, to_dir=output_dir + '/stable', meaningful_name=True)
        # 把分好类的不稳定阶段的图片存本地
        res.pick_and_save(unstable, 40, to_dir=output_dir + '/unstable', meaningful_name=True)

        # 对切分后的稳定区间，进行归类
        classify_result = cl.classify(file_name, stable, keep_data=True)
        result_dict = classify_result.to_dict()

        # 打印结果
        logger.info(f"参数组 {params} 的分类结果:")
        logger.info(result_dict.keys())
        logger.info(result_dict['0'][-1][-1])
        logger.info(result_dict)

        # 输出html报告
        report_name = f'./data/result_{video_name}_{threshold}_{offset}_{block}.html'
        r = Reporter()
        r.draw(classify_result, report_name)


# 移除默认的 DEBUG 级别控制台处理器
logger.remove()
# 添加 INFO 级别的控制台处理器
logger.add(sys.stderr, level="INFO")
# 添加 INFO 级别的文件处理器
logger.add("./data/reporter.log", level="INFO")

# 使用Keras方法进行预测
cl = KerasClassifier()
cl.load_model('./model/WzDemo1.weights.h5')

date = time.strftime('%Y-%m-%d', time.localtime(time.time()))

# 切割视频参数。 视频压缩率：compress_rate, 门限值：threshold, 偏移量：offset, block , date
# 调试参数配置
param_list = [[0.2, 0.90, 1, 2, date], [0.2, 0.91, 1, 2, date], [0.2, 0.92, 1, 2, date], [0.2, 0.93, 1, 2, date],
              [0.2, 0.94, 1, 2, date], [0.2, 0.95, 1, 2, date], [0.2, 0.96, 1, 2, date], [0.2, 0.97, 1, 2, date],
              [0.2, 0.98, 1, 2, date], [0.2, 0.99, 1, 2, date], [0.2, 1.00, 1, 2, date]]

# param_list = [[0.2, 0.97, 1, 6, date], [0.2, 0.97, 2, 6, date], [0.2, 0.97, 3, 6, date],
#               [0.2, 0.96, 1, 8, date], [0.2, 0.96, 2, 8, date], [0.2, 0.96, 3, 8, date],
#               [0.2, 0.96, 1, 6, date], [0.2, 0.96, 2, 6, date], [0.2, 0.96, 3, 6, date],
#               [0.2, 0.97, 1, 8, date], [0.2, 0.97, 2, 8, date], [0.2, 0.97, 3, 8, date],
#               [0.2, 0.97, 1, 4, date], [0.2, 0.97, 4, 8, date], [0.2, 0.96, 1, 4, date],
#               [0.2, 0.96, 1, 8, date]
#               ]

# 将视频切分成帧
file_name = './video/3.mp4'
# 获取视频名去掉后缀
video_name = file_name.split('/')[-1].split('.')[0]
video = VideoObject(file_name)
video.load_frames()

# 压缩视频
cutter = VideoCutter(compress_rate=0.2)

# 调用函数处理视频
process_video_with_params(file_name, param_list, cl, cutter)