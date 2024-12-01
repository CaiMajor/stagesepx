"""
这个例子描述了如何采集训练集
"""
from stagesepx.cutter import VideoCutter


video_path = '../../Vulakn打开_6分钟2024_10_18_14_42_IMG_9468.MOV'

# --- cut ---
cutter = VideoCutter()
res = cutter.cut(video_path)
stable, unstable = res.get_range()

res.pick_and_save(
    stable,
    # 每段区间的采样数，5即每个阶段等距离截取5张图片
    5,
    # 采样结果保存的位置
    './cut_result',
)
