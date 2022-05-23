import pandas as pd
import numpy as np
from gesture_rec.DynamicGestureRecognition.lib import image as kmg
from collections import deque

class DataLoader():
    """ 用于加载csv
    # Arguments
        path_vid    : path to the folder containing the extracted Jester dataset
        path_labels : path to the csv containing the labels
        path_train  : path to the csv containing the list of videos for training
        path_val    : path to the csv containing the list of videos used for validation
        path_test   : path to the csv containing the list of videos used for test
    # Returns
        An instance of the DataLoader class  
    """
    def __init__(self, path_vid, path_labels, path_train=None, path_val=None, path_test=None):
        self.path_vid    = path_vid
        self.path_labels = path_labels
        self.path_train  = path_train
        self.path_val    = path_val
        self.path_test   = path_test

        self.get_labels(path_labels)

        if self.path_train:
            self.train_df = self.load_video_labels(self.path_train)

        if self.path_val:
            self.val_df = self.load_video_labels(self.path_val)

        if self.path_test:
            self.test_df = self.load_video_labels(self.path_test, mode="input")

    def get_labels(self, path_labels):
        """从csv文件中加载标签数据，并创建 将标签映射为int型 和 将int型映射为标签的字典
        # Arguments
            path_labels : 包含标签的CSV文件的路径
        """
        # 使用pandas读取标签CSV
        self.labels_df = pd.read_csv(path_labels, names=['label'])
        # 以字符串形式把标签放到列表中  ['Swiping Left', 'Swiping Right',...]
        self.labels = [str(label[0]) for label in self.labels_df.values]
        # 获取标签列表的长度
        self.no_of_labels = len(self.labels)
        # 创建一个将标签映射为整数的字典 {'Swiping Left': 0, 'Swiping Right': 1,...}
        self.label_to_int = dict(zip(self.labels, range(self.no_of_labels)))
        #  创建一个将整数映射到标签的字典 {0: 'Swiping Left', 1: 'Swiping Right',..}
        self.int_to_label = dict(enumerate(self.labels))

    def load_video_labels(self, path_subset, mode="label"):
        """ 从csv文件中加载数据帧
        # Arguments
            path_subset : String, 要加载的CSV文件的路径
            mode        : String, (default: label), 如果模式设置为‘label’, filters rows if the labels exists in the labels Dataframe loaded previously
        # Returns
            A DataFrame
        """
        if mode=="input":
            names=['video_id']  #test 测试集没有 label
        elif mode=="label":
            names=['video_id', 'label']
        #seq:指定分割符，默认是’,’    names: 指定列名
        df = pd.read_csv(path_subset, sep=';', names=names) #df 存放有 id 和标签 的 Series
        
        if mode == "label":
            df = df[df.label.isin(self.labels)]  #保留标签中的数据

        return df
    
    def categorical_to_label(self, vector):
        """ Convert a vector to its associated string label
        # Arguments
            vector : Vector representing the label of a video
        # Returns
            Returns a String that is the label of a video
        """
        return self.int_to_label[np.where(vector==1)[0][0]]


class FrameQueue(object):
    """ Class used to create a queue from video frames
    # Arguments
        nb_frames   : no of frames for each video
        target_size : size of each frame
    # Returns
        A batch of frames as input for the model   作为模型输入的一批帧
    """
    def __init__(self, nb_frames, target_size): # 每个视频的帧数：16   输入的目标帧大小：(64,96)
        self.target_size = target_size #64，96
        self.nb_frames = nb_frames #16

        # 创建一个新的给定形状的数组，用 0 填充
        # representing the batch of frames  表示帧的批次
        #  batch : (1, 16, 64, 96, 3)
        self.batch = np.zeros((1, self.nb_frames) + target_size + (3,))

    def img_in_queue(self, img):
        # img : (240, 320, 3)
        # 遍历16张图片   for i in range(15)
        for i in range(self.batch.shape[1] - 1):   #0, 1,  ....  ,14
            self.batch[0, i] = self.batch[0, i+1]  #(64,96,3)
        # 加载图片并改变Img大小  (240, 320, 3) -> (64,96,3)
        img = kmg.load_img_from_array(img, target_size=self.target_size)

        # 将PIL图像实例转换为Numpy数组
        x = kmg.img_to_array(img)  #<class 'PIL.Image.Image'> (64, 96, 3)
        # 将图像归一化并添加到批处理中
        self.batch[0, self.batch.shape[1] - 1] = x / 255

        return self.batch

class FrameDeque(object):
    def __init__(self,target_size):
        self.target_size = target_size  # 64，96
        # 创建一个新的为空的双端队列
        self.img_deque = deque(maxlen=16)

    def img_in_queue(self, img):
        img = kmg.load_img_from_array(img, target_size=self.target_size)
        x = kmg.img_to_array(img)
        self.img_deque.append(img) # n , 64, 96, 3
        return self.img_deque