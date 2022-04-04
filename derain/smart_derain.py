# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets
from derain import Ui_DerainForm
from video_box import VideoBox
from util.utils import *
from settings import *
import os, time, train
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.reset_default_graph()

# 去雨菜单 继承derain.py
# 主要界面都在derain.py内完成
# 主要去雨的逻辑操作都在此.py文件内执行
class SmartDerainWindow(QtWidgets.QWidget, Ui_DerainForm):

    def __init__(self):
        super(SmartDerainWindow, self).__init__()
        self.setupUi(self)
        self.cwd = os.getcwd()  # 当前路径
        self.size = (256, 256)  # 所处理视频的分辨率，默认256*256
        self.fps = 10          # 视频帧率，默认10
        self.frame_count = 100    # 视频总帧数
        self.progressBar.setValue(0)
        self.openfile_btn.clicked.connect(self.open_video_file)
        self.video_box = VideoBox()
        self.video_box1 = VideoBox()
        self.video_compare_constrain.addWidget(self.video_box)
        self.video_compare_constrain.addWidget(self.video_box1)

    def open_video_file(self):
        # 文件打开窗口，路径默认，最后一个参数文件的过滤，不满足条件的不会显示
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开文件", self.cwd, "All Files (*);; Video File(*.mp4 *.flv)")
        if filename:
            print(f"file: {filename}")
            self.gener_video(filename)
            self.video_box.set_video(filename)

    def gener_video(self, input_path, video_name="output.mp4"):

        cap = cv2.VideoCapture(input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        weight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        size = (weight, height)
        self.fps = fps
        self.size = size
        self.frame_count = frame_count

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path+video_name, fourcc, fps, size)

        print(size)

        start = time.time()

        ret, rain1 = cap.read()
        rain1 = np.expand_dims(rain1, axis=0)
        rainy = tf.cast(rain1, tf.float32) / 255.0

        rain_pad = tf.pad(rainy, [[0, 0], [10, 10], [10, 10], [0, 0]], "SYMMETRIC")

        detail, base = train.inference(rain_pad)

        detail = detail[:, 6:tf.shape(detail)[1] - 6, 6:tf.shape(detail)[2] - 6, :]
        base = base[:, 10:tf.shape(base)[1] - 10, 10:tf.shape(base)[2] - 10, :]

        output = tf.clip_by_value(base + detail, 0., 1.)
        output = output[0, :, :, :]

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()

        with tf.Session(config=config) as sess:
            with tf.device('/gpu:0'):
                ckpt = tf.train.latest_checkpoint(model_path)  # try your own model
                saver.restore(sess, ckpt)
                print("Loading >>>>>>>>>>>>>")
                for i in range(20):   # 这里改为self.frame_count
                    derained, ori = sess.run([output, rainy])
                    derained = np.uint8(derained * 255.)
                    if derained.ndim == 3:
                        derained = derained[:, :, ::-1]  ### RGB to BGR
                    print(derained.shape)
                    video.write(derained)
                    #video.write(derained)
                    message = '{} / {} fps processed'.format(i + 1, self.frame_count)
                    print(message)
                    # self.process_label.setText(message)
                    self.progressBar.setValue(int((i + 1) / self.frame_count)*100)
        self.progressBar.setValue(100)
        sess.close()
        video.release()
        cap.release()
        cv2.destroyAllWindows()
        end = time.time() - start
        # print('视频生成完成')
        # print("用时{}秒".format(end))
        self.video_box1.set_video(video_path+video_name)
        print(video_path+video_name)
        self.process_label.setText('视频生成完成, 用时{}秒'.format(end))

    # 测试事件
    def msg(self):
        QtWidgets.QMessageBox.information(self, "标题", "测试用", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    smart = SmartDerainWindow()
    smart.show()
    sys.exit(app.exec_())