# smartcabin 作为无人驾驶的智慧座舱项目

# 1.环境
- 系统：windows, macos 或 linux
- IDE: pycharm 或 Qt(https://www.qt.io/download-qt-installer)
- python：3.8 或 3.9
- git
- pyqt5==5.15.2
# 2. 模块
- dehaze 去雾/霾/尘 
- derain 去雨 
- eye_track 眼动追踪
- face_rec 人脸识别
- gesture_rec 手势识别
- imglink 全景拼接
- semantic_seg 图像分割
# 3. 建议
- 项目的入口为 smartcabin.py
  run cmd : python smartcabin.py
- 项目的主窗口为 mainWindow.ui, 请暂时不要更改此文件
- 请在相应的目录下书写相应的功能，如非必须，尽量不去更改项目入口文件smartcabin.py
# 4. 用法
1. git clone https://github.com/FrankOcean/smartcabin.git
2. 创建自己的分支：(run cmd)
- 创建分支：git branch (branchname)
- 切换分支：git checkout (branchname)
- 合并分支: git merge 

    tips:（branchname）名称随意，自己知道就好
    eg: git branch derain
    最好自己相应模块功能全部实现后，再合并到主分支，不容易造成合并冲突

# 5. 有问题随时@我