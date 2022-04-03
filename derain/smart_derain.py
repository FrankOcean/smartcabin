# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets
from derain import Ui_DerainForm
from utils import *
import os, time, train
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.reset_default_graph()

model_path = './model/'
pre_trained_model_path = './model/trained/model'

img_path = './TestData/deraining_sequences/highway/input/'
results_path = './TestData/results/' # the path of de-rained images
video_path = './TestData/video_output/'

# 去雨菜单 继承derain.py
# 主要界面都在derain.py内完成
# 主要去雨的逻辑操作都在此.py文件内执行
class SmartDerainWindow(QtWidgets.QWidget, Ui_DerainForm):

    def __init__(self):
        super(SmartDerainWindow, self).__init__()
        self.setupUi(self)

    # 测试事件
    def msg(self):
        QtWidgets.QMessageBox.information(self, "标题", "测试用", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        rainy = tf.cast(image_decoded, tf.float32) / 255.0
        return rainy

    def open_video_file(self):
        self.msg()

    def start_handle_rainy_video(self):

        del_file("./tmp/")
        time.sleep(5)

    def gener_video(self, file_dir=results_path, video_name="output.mp4"):
        size = (256, 256)
        whole_path = read_from_file_k(file_dir)  # 图片全路径
        num_img = len(whole_path)
        for item in whole_path:
            if item.endswith('.png'):
                shape = cv2.imread(item).shape
                size = (shape[1], shape[0])
                break

        fps = 10  # 我设定位视频每秒1帧，可以自行修改
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path + video_name, fourcc, fps, size)

        start = time.time()

        filename_tensor = tf.convert_to_tensor(whole_path, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.prefetch(buffer_size=10)
        dataset = dataset.batch(batch_size=1).repeat()
        iterator = dataset.make_one_shot_iterator()

        rain = iterator.get_next()
        rain_pad = tf.pad(rain, [[0, 0], [10, 10], [10, 10], [0, 0]], "SYMMETRIC")

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
                if tf.train.get_checkpoint_state(model_path):
                    ckpt = tf.train.latest_checkpoint(model_path)  # try your own model
                    saver.restore(sess, ckpt)
                    print("Loading model")
                else:
                    saver.restore(sess, pre_trained_model_path)  # try a pre-trained model
                    print("Loading pre-trained model")

                for i in range(num_img):
                    derained, ori = sess.run([output, rain])
                    derained = np.uint8(derained * 255.)
                    if derained.ndim == 3:
                        derained = derained[:, :, ::-1]  ### RGB to BGR
                    video.write(derained)
                    print('%d / %d fps processed' % (i + 1, num_img))
        end = time.time() - start
        sess.close()
        video.release()
        cv2.destroyAllWindows()
        print('视频合成生成完成')
        print("用时{}秒".format(end))

    def test_whole_datasets(self):
        base_path = './data/input/'
        basefilenamelist = os.listdir(base_path)
        basefilenamelist = [f for f in basefilenamelist if not file_is_hidden(f)]  # 删除隐藏文件
        #     full_path = []
        for filename in basefilenamelist:
            input_filepath = base_path + filename + "/input/"  # './TestData/deraining_sequences/loveletter2/input/'
            #         full_path.append(input_filepath)
            print(input_filepath)
            self.gener_video(file_dir=input_filepath, video_name="output_" + filename + ".mp4")
            sync_video(file_dir=input_filepath, video_name="input_" + filename + ".mp4")
        print('全部处理完成')

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    smart = SmartDerainWindow()
    smart.show()
    sys.exit(app.exec_())