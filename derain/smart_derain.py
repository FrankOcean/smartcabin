# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets
import os, threading
import tensorflow.compat.v1 as tf

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
tf.disable_v2_behavior()
tf.reset_default_graph()
############################################################################

try:
    # run in smart_derain.py
    from derain import Ui_DerainForm
    from video_box import VideoBox
    from util.utils import *
    from opt import *
    import train
except ImportError:
    # run in smartcabin.py
    from derain.derain import Ui_DerainForm
    from derain.video_box import VideoBox
    from derain.util.utils import *
    from derain.opt import *
    import derain.train

# 去雨菜单 继承derain.py
# 主要界面都在derain.py内完成
# 主要去雨的逻辑操作都在此.py文件内执行
class SmartDerainWindow(QtWidgets.QWidget, Ui_DerainForm):

    def __init__(self):
        super(SmartDerainWindow, self).__init__()
        self.setupUi(self)
        self.cwd = os.getcwd()  # 当前路径
        self.size = (256, 256)  # 所处理视频的分辨率，默认256*256
        self.fps = 10           # 视频帧率，默认10
        self.frame_count = 100    # 视频总帧数
        self.progressBar.setValue(0)
        self.openfile_btn.clicked.connect(self.open_video_file)
        self.video_box = VideoBox()
        self.video_box1 = VideoBox()
        self.video_compare_constrain.addWidget(self.video_box)
        self.video_compare_constrain.addWidget(self.video_box1)

    def open_video_file(self):
        # 文件打开窗口，路径默认，最后一个参数文件的过滤，不满足条件的不会显示
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开文件", self.cwd, "Video File(*.mp4 *.flv);;All Files (*)")
        if filepath:
            print(f"file: {filepath}")
            filename = filepath.split('/')[-1].split(".")[0]
            video_name = "out_" + filename + ".mp4"
            print(video_name)
            #self.gener_video(filepath, video_name=video_name)
            self.video_box.set_video(filepath)
            try:
                t = threading.Thread(target=self.gener_video, args=(filepath, video_name,))
                t.start()
            except:
                print("Error: unable to start thread")

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

        start = time.time()
        ret = True
        rainy_arr = []
        while ret:
            ret, ori = cap.read()
            if ret:
                ori_nor = tf.cast(ori, tf.float32) / 255.0
                rainy_arr.append(ori_nor)


        dataset = tf.data.Dataset.from_tensor_slices((rainy_arr))
        dataset = dataset.prefetch(buffer_size=10)
        dataset = dataset.batch(batch_size=1).repeat()
        iterator = dataset.make_one_shot_iterator()
        rainy = iterator.get_next()

        rain_pad = tf.pad(rainy, [[0, 0], [10, 10], [10, 10], [0, 0]], "SYMMETRIC")

        detail, base = train.inference(rain_pad)

        detail = detail[:, 6:tf.shape(detail)[1] - 6, 6:tf.shape(detail)[2] - 6, :]
        base = base[:, 10:tf.shape(base)[1] - 10, 10:tf.shape(base)[2] - 10, :]

        output = tf.clip_by_value(base + detail, 0., 1.)
        output = output[0, :, :, :]

        message = "视频帧解析完成，分辨率{}，帧率{}，帧数{}, 用时{:.2f}".format(size, fps, frame_count, time.time()-start)
        self.process_label.setText(message)
        time.sleep(3)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        saver = tf.train.Saver()

        start = time.time()
        with tf.Session(config=config) as sess:
            with tf.device('/gpu:0'):
                ckpt = tf.train.latest_checkpoint(model_path)  # try your own model
                saver.restore(sess, ckpt)
                print("Loading >>>>>>>>>>>>>")
                for i in range(10):   # 这里改为frame_count
                    derained, ori = sess.run([output, rainy])
                    derained = np.uint8(derained * 255.)
                    if derained.ndim == 3:
                        derained = derained[:, :, ::-1]  ### RGB to BGR
                    video.write(derained)
                    message = '第 {} / {} 帧去雨中'.format(i + 1, frame_count)
                    print(message)
                    self.process_label.setText(message)
                    self.progressBar.setValue(int(i/frame_count)*1000)
        # self.progressBar.setValue(100)
        sess.close()
        video.release()
        cap.release()
        cv2.destroyAllWindows()
        end = time.time() - start
        # print('视频生成完成')
        # print("用时{}秒".format(end))
        print(video_path+video_name)
        self.process_label.setText('视频生成完成, 用时{:.2f}秒'.format(end))
        time.sleep(5)
        self.video_box1.set_video(video_path+video_name)

    # 测试事件
    def msg(self):
        QtWidgets.QMessageBox.information(self, "标题", "测试用", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    smart = SmartDerainWindow()
    smart.show()
    sys.exit(app.exec_())