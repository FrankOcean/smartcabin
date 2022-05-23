import os, sys
import torchvision.ops
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_file, non_max_suppression
from utils.torch_utils import select_device
from models.yolo import Model
from models.experimental import attempt_load
import cv2
import numpy as np
import pdb
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
def load_split_nvgesture(file_with_split = 'D:\gesture_recognition_datasets\data_set/nvGestures/nvgesture_test_correct_cvpr2016_v2.lst',list_split = list()):
    params_dictionary = dict()
    with open(file_with_split,'r') as f:
        dict_name = file_with_split[file_with_split.rfind('/')+1 :]
        dict_name = dict_name[:dict_name.find('_')]
        for line in f:
            params = line.split()
            #print(params)
            params_dictionary = dict()
            params_dictionary['dataset'] = dict_name
            path = params[0].split(':')[1]
            #print(path)
            for param in params[1:]:
                    parsed = param.split(':')
                    key = parsed[0]
                    if key == 'label':
                        # make label start from 0
                        label = int(parsed[1]) - 1 
                        params_dictionary['label'] = label
                    elif key in ('depth','color','duo_left'):
                        #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                        sensor_name = key
                        #first store path
                        params_dictionary[key] = path + '/' + parsed[1]
                        #store start frame
                        params_dictionary[key+'_start'] = int(parsed[2])
                        params_dictionary[key+'_end'] = int(parsed[3])
        
            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']                  

            list_split.append(params_dictionary)
 
    return list_split

def load_data_from_file(example_config, sensor,image_width, image_height):
    path = example_config[sensor] + ".avi"  #color视频路径 ./Video_data/class_01/subject3_r0/sk_color.avi
    start_frame = example_config[sensor+'_start'] #146
    end_frame = example_config[sensor+'_end'] #226
    label = example_config['label'] #0
    frames_to_load = range(start_frame, end_frame) #共80帧
    chnum = 3 if sensor == "color" else 1 #3
    video_container = np.zeros((image_height, image_width, chnum, 80), dtype = np.uint8) #(240,320,3,80)
    cap = cv2.VideoCapture(path)
    ret = 1
    frNum = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #设置视频帧数从start_frame这里开始
    for indx, frameIndx in enumerate(frames_to_load):    #indx 0-79  frameIndx 146-226
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(image_width, image_height)) #改变图像大小为 320,240
            if sensor != "color":
                frame = frame[...,0]
                frame = frame[...,np.newaxis]
            video_container[..., indx] = frame
        else:
            print("Could not load frame")
    cap.release()
    return video_container, label  #返回 80 张关键帧图像列表 及标签
class _3DMobileNetV1(nn.Module):
    def __init__(self):
        super(_3DMobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):  # 第一层传统的卷积：conv3*3+BN+ReLU
            return nn.Sequential(
                nn.Conv3d(inp, oup, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), dilation=(1, 1, 1),
                          bias=False),
                nn.BatchNorm3d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):  # 其它层的depthwise convolution：conv3*3+BN+ReLU+conv1*1+BN+ReLU
            return nn.Sequential(
                nn.Conv3d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
                # groups参数把输入通道进行分组，默认为1不分组
                nn.BatchNorm3d(inp),
                nn.ReLU(inplace=True),

                nn.Conv3d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(oup),
                nn.ReLU(inplace=True),
            )

        self.first = nn.Sequential(conv_bn(3, 32, (1, 2, 2)))  # 第一层传统的卷积
        self.model = nn.Sequential(
            conv_dw(32, 64, 1),  # 其它层depthwise convolution
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 2),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1)
        )
        self.pool = nn.AvgPool3d((3, 4, 4))
        self.fc = nn.Linear(1024, 25)  # 全连接层

    def forward(self, x):
        #print(x.shape)
        x = self.first(x)
        #print(x.shape)
        x = self.model(x)
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = x.view(-1, 1024)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x



sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
file_lists = dict()
file_lists["test"] = "./nvgesture_test_correct_cvpr2016_v2.lst"
file_lists["train"] = "./nvgesture_train_correct_cvpr2016_v2.lst"
train_list = list()
test_list = list()
load_split_nvgesture(file_with_split = file_lists["train"],list_split = train_list)
load_split_nvgesture(file_with_split = file_lists["test"],list_split = test_list)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = _3DMobileNetV1().to(device=device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 25], gamma=0.1)
epochs=10
loss_y = []
best_acc = 0
device = select_device("0")
# Create model
yolo_model = attempt_load("./weights/yolov3_dect_all_hand.pt", map_location=device)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i in range(len(train_list)):
        data, label = load_data_from_file(example_config = train_list[i], sensor = sensors[0], image_width = 320, image_height = 240)
        data=data.transpose((3, 2, 0, 1))
        #yolo
        for img in data:
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            resize = transforms.Resize((416,416))
            img = resize(img)
            pred = yolo_model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred)
            #x y x y 准确率 分类
            pred = pred[0]
            print(pred)

        img_dstack=torch.permute(torch.from_numpy(data), (2, 3, 0,1))
        transform=transforms.Resize((224,224))
        img_dstack=transform(img_dstack)
        img_dstack = img_dstack.view((1, 3, 80, 224, 224))
        img_dstack=torch.tensor(img_dstack,dtype=torch.float)
        #print(img_dstack.shape)
        model.train()
        optimizer.zero_grad()
        logits = model(img_dstack.to(device))
        #print(logits)
        #print(logits.shape)
        # loss0 = loss_function(logits, torch.tensor([labels[0],labels[30]]).reshape((2)).to(device))
        loss0 = loss_function(logits, torch.tensor([label]).reshape((1)).to(device))
        loss = loss0
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f'[{i + 1}]/[{len(train_list)}]  loss为{loss}')















    '''
    -------print(train_list[0])
    {'dataset': 'nvgesture', 'depth': './Video_data/class_01/subject3_r0/sk_depth',
    'depth_start': 146, 'depth_end': 226, 'color': './Video_data/class_01/subject3_r0/sk_color',
    'color_start': 146, 'color_end': 226, 'duo_left': './Video_data/class_01/subject3_r0/duo_left',
    'duo_left_start': 170, 'duo_left_end': 263, 'label': 0,
    'duo_right': './Video_data/class_01/subject3_r0/duo_right', 'duo_right_start': 170,
    'duo_right_end': 263, 'duo_disparity': './Video_data/class_01/subject3_r0/duo_disparity',
    'duo_disparity_start': 170, 'duo_disparity_end': 263}
    '''
    #pdb.set_trace()#pdb是python用于调试代码的常用库如果在代码中看到pdb.set_trace()程序运行到这里就会暂停。