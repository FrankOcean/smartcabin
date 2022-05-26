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
import lib.model as model
from lib.HandTrackingModule import handDetector
from lib.data_loader import FrameQueue,FrameDeque
import time
#运行：   python hanuman.py --config "gesture_control_config.cfg"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

#   函数加载预训练的CNN模型
def load_model(model_json_path, model_weights_path):
    # read the model json file  ../models/radhakrishna.json
    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # load the model from json file  从json文件中加载模型
    #net = tf.keras.models.model_from_json(loaded_model_json) # resnet 和原文
    net = model.CNN3D_lite(inp_shape=(16,64,96,3), nb_classes=5) #cnn
    # load weights into new model  ../models/radhakrishna.h5
    net.load_weights(model_weights_path)
    print("Loaded CNN model from disk")
    return net

def main(args):
    # 从配置文件中提取信息
    nb_frames                = config.getint('general', 'nb_frames')
    target_size              = literal_eval(config.get('general', 'target_size'))
    nb_classes               = config.getint('general', 'nb_classes')
    csv_labels               = config.get('path', 'csv_labels')
    gesture_keyboard_mapping = config.get('path', 'gesture_keyboard_mapping')
    model_json_path          = config.get('path', 'model_json_path')
    model_weights_path       = config.get('path', 'model_weights_path')

    # 打开标签文件
    with open(csv_labels)as f:
        f_csv = csv.reader(f)
        # 创建空列表来存储标签列表
        labels_list = []
        for row in f_csv:
            labels_list.append(row)
        # 将标签列表转换为元组
        labels_list = tuple(labels_list)

    #  从配置加载手势->键映射
    mapping = configparser.ConfigParser()
    action = {}
    # 如果指定的映射文件存在  mapping.ini
    if os.path.isfile(gesture_keyboard_mapping):
        # read the file
        mapping.read(gesture_keyboard_mapping)
        # for each mapping in the mapping file
        for m in mapping['MAPPING']:
            # m为 = 左边的  thumb up 、 thumb down 等等
            # mapping['MAPPING'][m] 为 = 右边的值  press,space
            #val 为 装着 ['press', 'space'] 的列表
            val = mapping['MAPPING'][m].split(',')
            # 设置操作
            action[m] = {'fn': val[0], 'keys': val[1:]}  # fn:  press ： 按下,cmd:命令提示符   typewrite：打字
            #action 格式  ：  {'thumb up': {'fn': 'press', 'keys': ['space']},......}
    else:
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), args.mapping)

    # 创建一个deque（双向队列）来存储识别的手势，队列最大长度为3，超过3原队列元素被挤出
    act = deque(['No gesture', "No gesture"], maxlen=4)
    # 加载网络模型及其权重
    net = load_model(model_json_path, model_weights_path)
    # 初始化帧队列
    frame_queue = FrameQueue(nb_frames, target_size) # 每个视频的帧数：16   输入的目标帧大小：(64,96)

    # 预加载网络
    yu_batch = np.zeros((1, 16) + (64,96) + (3,))
    net.predict(yu_batch)

    print("预加载网络----------------------------------------------------")
    # 加载手部检测跟踪
    hand_tracking = handDetector()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 设置图像的长宽
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    frame_num = 0
    batch_x= []
    t0=0.00
    t1=0.00
    gesture='No gesture'
    Acc=0.0

    while(cap.isOpened()):
        on, frame = cap.read()
        frame = cv2.flip(frame, 1)
        flag = hand_tracking.findHands(frame)
        frame = cv2.flip(frame, 1)
        #print(flag)  #判断是否出现右手，返回值类型为 bool
        if flag:
            t0 = time.perf_counter()
            frame_num += 1
            #把读入的 bgr模式转换为 rgb
            b, g, r = cv2.split(frame)
            frame_calibrated = cv2.merge([r, g, b])  # (240, 320, 3)
            if frame_num < 16:
                # 输入模型的 batch 大小  (1, 16, 64, 96, 3)
                batch_x = frame_queue.img_in_queue(frame_calibrated)
                frame = cv2.resize(frame, (672, 500))
                frame = cv2.putText(frame, text=f'class: {gesture}', org=(10, 50),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
                frame = cv2.putText(frame, text=' Acc: {:.5}%'.format(Acc), org=(10, 100),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
                cv2.imshow('camera0', frame)
                if cv2.waitKey(30) & 0xFF == 27:
                    break
            else:
                # 预测
                res = net.predict(batch_x,use_multiprocessing=True)
                # 分类
                predicted_class = labels_list[np.argmax(res)]
                #print('Predicted Class = ', predicted_class, 'Accuracy = ', np.amax(res)*100,'%')
                # 如果结果的最大概率大于阈值，将手势设置为预测标签，else设置为“No Gesture”
                gesture = (labels_list[np.argmax(res)] if max(res[0]) > 0.85 else labels_list[4])[0]
                # print(gesture)
                # 将手势转换为小写
                # gesture = gesture.lower()
                Acc=np.amax(res)*100
                print('Predicted Class = ', gesture, 'Accuracy = ', Acc, '%')

                # 如果动作队列中的第一个手势与第二个手势不一样，并且队列中不重复的元素个数为1
                """
                设置交互操作
                """
                #清空
                frame_num = 0
                batch_x=[]

                frame=cv2.resize(frame,(672,500))
                if gesture=='No gesture':
                    Acc=0.000

                frame = cv2.putText(frame, text=f'class: {gesture}', org=(10, 50),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
                frame = cv2.putText(frame, text=' Acc: {:.5}%'.format(Acc), org=(10, 100),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
                cv2.imshow('camera0', frame)
                if cv2.waitKey(30) & 0xFF==27:
                    break
        else:
            t1 = time.perf_counter()
            if t1-t0>2:
                frame_num = 0
                batch_x = []
                #print("---------手势列表已重置------------")

            frame = cv2.resize(frame, (672, 500))
            frame = cv2.putText(frame, text=f'class: {gesture}', org=(10, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
            frame = cv2.putText(frame, text=' Acc: {:.5}%'.format(Acc), org=(10, 100),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=3)
            cv2.imshow('camera0', frame)
            if cv2.waitKey(30) & 0xFF==27:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()#参数对象
    parser.add_argument("--config",dest="config", default="./test123123.cfg",help="运行脚本所需的配置文件")#添加参数
    args = parser.parse_args()#解析参数
    config = configparser.ConfigParser()
    config.read(args.config)
    main(config)








