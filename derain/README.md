# derain 作为智慧座舱项目的一个分支功能

# 1.环境
- tensorflow==2.5.0
- opencv-python==4.5.0
- numpy==1.19.5
- PyQt5==5.15.2
# 2. 说明
- 从smartcabin中启动，需要把derain文件夹make derectory as sources root 
- opt.py 为全局配置文件
- GuidedFilter 为可训练的过滤器，具体参考：https://github.com/wuhuikai/DeepGuidedFilter  
- smart_derain.py 可单独运行，也可以做为模块运行
- video_box.py 可单独运行，作为单独的视频播放器
usage: VideoBox().set_video("video_path", VideoBox.VIDEO_TYPE_OFFLINE, False)
- 因后续会有较大改动，分支中暂不提供requirements.txt文件，所需三方库请自行下载