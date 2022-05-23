# derain
import os

cwd = os.getcwd()
model_path = cwd + '/model/'                # 模型文件存放路径
video_path = cwd + '/data/output/'          # derain后视频存放路径

if not os.path.exists(model_path):
    model_path = cwd + '/derain/model/'
    video_path = cwd + '/derain/data/output/'
