"""
这是一个最小化的 stagesepx 使用例子
每一行的注释均可以在 cut_and_classify.py 中找到
"""
from stagesepx.cutter import VideoCutter
from stagesepx.classifier import SVMClassifier
from stagesepx.reporter import Reporter
from stagesepx.video import VideoObject

# video_path = "../demo.mp4"
# video_path = '../Vulakn打开_6分钟2024_10_18_14_42_IMG_9468.MOV'
# video_path = '../Vulakn关闭_6分钟_2024_10_18_14_22_IMG_9466.MOV'
# video_path = '../Vulakn关闭_3分钟_2024_10_18_14_15_IMG_9465.MOV'
video_path = '../Vulakn打开_3分钟_2024_10_18_14_32_IMG_9467.MOV'

video = VideoObject(video_path)
video.load_frames()

# --- cutter ---
cutter = VideoCutter()
res = cutter.cut(video)
stable, unstable = res.get_range()
data_home = res.pick_and_save(stable, 5)

# --- classify ---
cl = SVMClassifier()
cl.load(data_home)
cl.train()
classify_result = cl.classify(video, stable)

# --- draw ---
r = Reporter()
r.draw(classify_result)
