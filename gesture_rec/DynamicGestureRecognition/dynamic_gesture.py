import cv2
import os
import sys
import csv
import numpy as np
from collections import deque
import pyautogui
import argparse
import configparser
from ast import literal_eval
import errno
import keras
import tensorflow as tf
from gesture_rec.DynamicGestureRecognition.lib.HandTrackingModule import handDetector
from gesture_rec.DynamicGestureRecognition.lib.data_loader import FrameQueue,FrameDeque
from gesture_rec.DynamicGestureRecognition.lib import model
import time

class Dynamic_gesture():
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser()  # 参数对象
        parser.add_argument("--config", dest="config", default="./gesture_rec/DynamicGestureRecognition/My_gesture_control_config.cfg",help="运行脚本所需的配置文件")  # 添加参数
        #parser.add_argument("--config", dest="config", default="./test123123.cfg",help="运行脚本所需的配置文件")  # 添加参数
        args = parser.parse_args()  # 解析参数
        self.config = configparser.ConfigParser()
        self.config.read(args.config)
        # 从配置文件中提取信息
        self.nb_frames = self.config.getint('general', 'nb_frames')
        self.target_size = literal_eval(self.config.get('general', 'target_size'))
        self.nb_classes = self.config.getint('general', 'nb_classes')
        self.csv_labels = self.config.get('path', 'csv_labels')
        self.gesture_keyboard_mapping = self.config.get('path', 'gesture_keyboard_mapping')
        self.model_json_path = self.config.get('path', 'model_json_path')
        self.model_weights_path = self.config.get('path', 'model_weights_path')
        
        # 创建一个deque（双向队列）来存储识别的手势，队列最大长度为3，超过3原队列元素被挤出
        self.act = deque(['No self.gesture', "No self.gesture"], maxlen=4)
        # 加载网络模型及其权重
        self.net = Dynamic_gesture.load_model(self.model_json_path, self.model_weights_path)
        # 初始化帧队列
        self.frame_queue = FrameQueue(self.nb_frames, self.target_size)  # 每个视频的帧数：16   输入的目标帧大小：(64,96)
        # 预加载网络
        yu_batch = np.zeros((1, 16) + (64, 96) + (3,))
        self.net.predict(yu_batch)
        #print("预加载网络----------------------------------------------------")

        #初始参数
        self.frame_num = 0
        self.batch_x = []
        self.t0 = 0.00
        self.t1 = 0.00
        self.gesture = 'No self.gesture'
        self.Acc = 0.0
        self.action = {} #映射集合
        self.labels_list = Dynamic_gesture.csv_label(self) # 创建空列表来存储标签列表

    @staticmethod
    def load_model(model_json_path, model_weights_path):
        #函数加载预训练的CNN模型
        # read the model json file  ../models/radhakrishna.json
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # load the model from json file  从json文件中加载模型
        net = model.CNN3D_lite(inp_shape=(16,64,96,3), nb_classes=5) #cnn
        # load weights into new model  ../models/radhakrishna.h5
        net.load_weights(model_weights_path)
        #print("Loaded CNN model from disk")
        return net

    @staticmethod
    def csv_label(self):
        # 打开标签文件
        with open(self.csv_labels)as f:
            labels_list = []
            f_csv = csv.reader(f)
            for row in f_csv:
                labels_list.append(row)
            # 将标签列表转换为元组
            labels_list = tuple(labels_list)
        return labels_list

    @staticmethod
    def mapping(self):
        #  从配置加载手势->键映射
        mapping = configparser.ConfigParser()
        # 如果指定的映射文件存在  mapping.ini
        if os.path.isfile(self.self.gesture_keyboard_mapping):
            # read the file
            mapping.read(self.self.gesture_keyboard_mapping)
            # for each mapping in the mapping file
            for m in mapping['MAPPING']:
                val = mapping['MAPPING'][m].split(',')
                # 设置操作
                self.action[m] = {'fn': val[0], 'keys': val[1:]}
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.config.mapping)
myNet = Dynamic_gesture()
hand_tracking = handDetector()

def dynamic_gesture(frame):
    frame = cv2.flip(frame, 1)
    flag = hand_tracking.findHands(frame)
    frame = cv2.flip(frame, 1)
    # print(flag)  #判断是否出现右手，返回值类型为 bool
    if flag:
        myNet.t0 = time.perf_counter()
        myNet.frame_num += 1
        # 把读入的 bgr模式转换为 rgb
        b, g, r = cv2.split(frame)
        frame_calibrated = cv2.merge([r, g, b])  # (240, 320, 3)
        if myNet.frame_num < 16:
            # 输入模型的 batch 大小  (1, 16, 64, 96, 3)
            myNet.batch_x = myNet.frame_queue.img_in_queue(frame_calibrated)
            frame = cv2.resize(frame, (672, 500))
            frame = cv2.flip(frame,1)
            frame = cv2.putText(frame, text=f'class: {myNet.gesture}', org=(10, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
            frame = cv2.putText(frame, text=' self.Acc: {:.5}%'.format(myNet.Acc), org=(10, 100),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
            return frame
        else:
            # 预测
            res = myNet.net.predict(myNet.batch_x)
            #print("-=-=-=-=",myNet.labels_list)
            # 分类
            predicted_class = myNet.labels_list[np.argmax(res)]
            # print('Predicted Class = ', predicted_class, 'self.Accuracy = ', np.amax(res)*100,'%')
            # 如果结果的最大概率大于阈值，将手势设置为预测标签，else设置为“No self.gesture”
            myNet.gesture = (myNet.labels_list[np.argmax(res)] if max(res[0]) > 0.85 else myNet.labels_list[4])[0]
            # print(self.gesture)
            # 将手势转换为小写
            # self.gesture = self.gesture.lower()
            myNet.Acc = np.amax(res) * 100
            print('Predicted Class = ', myNet.gesture, 'self.Accuracy = ', myNet.Acc, '%')
            # 如果动作队列中的第一个手势与第二个手势不一样，并且队列中不重复的元素个数为1
            """
            设置交互操作
            """
            # 清空
            myNet.frame_num = 0
            myNet.batch_x = []

            frame = cv2.resize(frame, (672, 500))
            if myNet.gesture == 'No self.gesture':
                myNet.Acc = 0.000
            frame = cv2.flip(frame, 1)
            frame = cv2.putText(frame, text=f'class: {myNet.gesture}', org=(10, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
            frame = cv2.putText(frame, text=' self.Acc: {:.5}%'.format(myNet.Acc), org=(10, 100),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
            return frame
    else:
        myNet.t1 = time.perf_counter()
        if myNet.t1 - myNet.t0 > 2:
            myNet.frame_num = 0
            myNet.batch_x = []
            # print("---------手势列表已重置------------")
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (672, 500))
        frame = cv2.putText(frame, text=f'class: {myNet.gesture}', org=(10, 50),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
        frame = cv2.putText(frame, text=' self.Acc: {:.5}%'.format(myNet.Acc), org=(10, 100),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
        return frame


if __name__ == '__main__':
    myNet = Dynamic_gesture()
    hand_tracking = handDetector()
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 设置图像的长宽
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while (cap.isOpened()):
        on, frame = cap.read()
        frame = dynamic_gesture(frame)
        cv2.imshow('camera0', frame)
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()




