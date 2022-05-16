import cv2, os, sys
import cv2.cv2
import torch
import torchvision.ops

current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory + os.path.sep + 'yolov5')
sys.path.append(current_directory + os.path.sep + 'yolov5')
sys.path.append(current_directory)
from torchvision import transforms

from ITrackerModel import ITrackerModel
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_file, non_max_suppression
from utils.torch_utils import select_device
from models.yolo import Model
from models.experimental import attempt_load
import numpy as np
import tkinter as tk

# w 29.65cm，h 52.71cm
cur = cv2.imread("../sq40.jpg", cv2.IMREAD_UNCHANGED)
#ppcm = 1920 / 52.71
ppcmw = 1920 / 28
ppcmh = 1080 / 13
alpha = 1.2
cap = cv2.VideoCapture(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"



model = attempt_load(current_directory+"/yolov5/models/best.pt", map_location=device)
dataset = LoadStreams("0")
tracker = ITrackerModel()
tracker.load_state_dict(torch.load(current_directory+"/checkpoint.pth.tar", map_location='cpu')['state_dict'])
tracker.to(device)

def process_img(img):
    img0 = img.copy()
    if cv2.waitKey(1) == ord('q'):  # q to quit
        cv2.destroyAllWindows()
        raise StopIteration

    # Letterbox
    img = [letterbox(x)[0] for x in img0]

    # Stack
    img = np.stack(img, 0)

    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    return img, img0

def pred_once(img, im0):
    im0 = im0[0]
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred)
    pred = pred[0]
    # print(pred)
    i = torchvision.ops.nms(pred[:, 0:4], pred[:, 4], 0.5)
    pred = pred[i]

    # pfinal = []
    # for cls in range(3):
    #     mconf = 0
    #     pos = -1
    #     for index, p in enumerate(pred):
    #         if cls == p[-1] and mconf < p[-2]:
    #             pos = index
    #     if pos != -1:
    #         pfinal.append(pred[pos])
    # print(f"predlen={len(pred)}, pfinallen={len(pfinal)}")
    # 假设左侧的为左眼，只有正注视的图像
    # print(pred)
    # pred = pred.tolist()
    # print(pred)
    predf = [None] * 3
    for x, y, w, h, ac, cls in pred:
        if cls == 0:
            predf[0] = [x, y, w, h, ac, cls]

    for x, y, w, h, ac, cls in pred:
        if cls != 0:
            if predf[1] is None:
                predf[1] = [x, y, w, h, ac, 1]
            elif x < predf[1][0]:
                predf[1] = [x, y, w, h, ac, 1]

    for x, y, w, h, ac, cls in pred:
        if cls != 0:
            if predf[2] is None:
                predf[2] = [x, y, w, h, ac, 2]
            elif x > predf[2][0]:
                predf[2] = [x, y, w, h, ac, 2]
    # cv2.imshow('frame1', im0)
    for p in predf:
        if p is None:
            continue
        x, y, w, h, ac, cls = [int(t) for t in p]
        if cls == 0:
            color = (255, 0, 0)
        elif cls == 1:
            color = (0, 255, 0)
        elif cls == 2:
            color = (0, 0, 255)
        print(x,y,w,h)
        im0 = cv2.rectangle(im0, (x, y), (w, h), color, 2)

    # cv2.imshow('frame1', im0)

    if None in predf:
        return [im0]
    # print(img.shape)
    # print(im0.shape)
    face_xyxy = [int(x) for x in predf[0][0:4]]
    eye1_xyxy = [int(x) for x in predf[1][0:4]]
    eye2_xyxy = [int(x) for x in predf[2][0:4]]
    # print("face_xyxy=",face_xyxy)

    # img = img.flip(dims=[3])
    # face = img[0, :, face_xyxy[0]:face_xyxy[2], face_xyxy[1]:face_xyxy[3]]
    # eye1为右眼，eye2为左眼
    face = img[:, :, face_xyxy[1]:face_xyxy[3], face_xyxy[0]:face_xyxy[2]]
    eye1 = img[:, :, eye1_xyxy[1]:eye1_xyxy[3], eye1_xyxy[0]:eye1_xyxy[2]]
    eye2 = img[:, :, eye2_xyxy[1]:eye2_xyxy[3], eye2_xyxy[0]:eye2_xyxy[2]]
    reSize = transforms.Resize([224, 224])
    face = reSize(face)
    eye1 = reSize(eye1)
    eye2 = reSize(eye2)
    print(face.shape, eye1.shape, eye2.shape)
    # face = torch.FloatTensor(face)
    # eye1 = torch.FloatTensor(eye1)
    # eye2 = torch.FloatTensor(eye2)
    # print(img[0])
    # print("faceshape=",face.shape)
    # toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    # pic1 = toPIL(img[0])
    # pic2 = toPIL(eye1)

    ylen, xlen = img[0][0].shape
    print(xlen, ylen, img[0].shape)
    xyxy = [face_xyxy[0] / xlen * 25, face_xyxy[1] / ylen * 25, face_xyxy[2] / xlen * 25, face_xyxy[3] / ylen * 25]
    print(xyxy)

    faceGrid = makeGrid(xyxy)
    faceGrid = torch.FloatTensor(faceGrid).to(device)
    faceGrid = faceGrid.reshape(1, 25, 25)
    # print(faceGrid.reshape(25, 25))

    gaze = tracker(face, eye2, eye1, faceGrid)
    print(gaze)
    x = gaze[0][0].item() * ppcmw
    y = gaze[0][1].item() * ppcmh

    xx = int((x + 1920 // 2))
    yy = int(y)
    print(xx, -yy)
    return [xx, -yy, im0]


def makeGrid(xyxy,gridSize=(25, 25)):
    gridLen = gridSize[0] * gridSize[1]
    grid = np.zeros([gridLen, ], np.float32)

    indsY = np.array([i // gridSize[0] for i in range(gridLen)])
    indsX = np.array([i % gridSize[0] for i in range(gridLen)])
    condX = np.logical_and(indsX >= xyxy[0], indsX < xyxy[2])
    condY = np.logical_and(indsY >= xyxy[1], indsY < xyxy[3])
    cond = np.logical_and(condX, condY)
    grid[cond] = 1
    return grid


def test1():
    cfg = "yolov5/models/yolov5s.yaml"
    cfg = check_file(cfg)  # check file
    device = select_device("0")

    # Create model
    model = attempt_load("yolov5/models/best.pt", map_location=device)
    # if True:

    while True:
        ret, frame = cap.read()
        # frame = cv2.imread("yolov5/data/detectdata/images/10.jpg")
        img = frame

        # xp,yp = frame.shape[0], frame.shape[1]

        frame = np.transpose(frame, (2, 0, 1))
        frame = frame / 255
        frame = torch.from_numpy(frame)
        frame = torch.unsqueeze(frame, 0)
        # print(frame.shape)
        frame = frame.float()
        frame = frame.to(device)
        pred = model(frame, augment=True)[0]
        # pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, agnostic=True)
        pred = non_max_suppression(pred, conf_thres=0.7)
        print(pred)

        for pr in pred:
            for p in pr:
                # print(p)
                if len(p) == 0:
                    continue
                # print(type(p))

                x, y, w, h, con, cls = [int(t.item()) for t in p]
                # print(x,y,w,h)
                # x *= xp
                # y *= yp
                # w *= xp
                # h *= yp
                x, y, w, h, ac, cls = [int(t) for t in p]
                if cls == 0:
                    color = (255, 0, 0)
                elif cls == 1:
                    color = (0, 255, 0)
                elif cls == 2:
                    color = (0, 0, 255)
                im0 = cv2.rectangle(im0, (x, y), (w, h), color, 2)

        cv2.imshow('frame2', img)
        # cv2.waitKey(10000)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


def test2():
    device = select_device("cpu")
    # Create model
    model = attempt_load("yolov5/models/best.pt", map_location=device)
    dataset = LoadStreams("0")
    tracker = ITrackerModel()

    tracker.load_state_dict(torch.load("checkpoint.pth.tar", map_location='cpu')['state_dict'])
    tracker.to(device)

    for path, img, im0s, vid_cap in dataset:
        # print(img[0].shape, im0s[0].shape)
        im0 = im0s[0].copy()
        # cv2.imshow('frame2', im0s[0])
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred)
        pred = pred[0]
        # print(pred)
        i = torchvision.ops.nms(pred[:, 0:4], pred[:, 4], 0.5)
        pred = pred[i]

        # pfinal = []
        # for cls in range(3):
        #     mconf = 0
        #     pos = -1
        #     for index, p in enumerate(pred):
        #         if cls == p[-1] and mconf < p[-2]:
        #             pos = index
        #     if pos != -1:
        #         pfinal.append(pred[pos])
        # print(f"predlen={len(pred)}, pfinallen={len(pfinal)}")
        # 假设左侧的为左眼，只有正注视的图像
        # print(pred)
        # pred = pred.tolist()
        # print(pred)
        predf = [None] * 3
        for x, y, w, h, ac, cls in pred:
            if cls == 0:
                predf[0] = [x, y, w, h, ac, cls]

        for x, y, w, h, ac, cls in pred:
            if cls != 0:
                if predf[1] is None:
                    predf[1] = [x, y, w, h, ac, 1]
                elif x < predf[1][0]:
                    predf[1] = [x, y, w, h, ac, 1]

        for x, y, w, h, ac, cls in pred:
            if cls != 0:
                if predf[2] is None:
                    predf[2] = [x, y, w, h, ac, 2]
                elif x > predf[2][0]:
                    predf[2] = [x, y, w, h, ac, 2]
        # cv2.imshow('frame1', im0)
        for p in predf:
            if p is None:
                continue
            x, y, w, h, ac, cls = [int(t) for t in p]
            if cls == 0:
                color = (255, 0, 0)
            elif cls == 1:
                color = (0, 255, 0)
            elif cls == 2:
                color = (0, 0, 255)
            im0 = cv2.rectangle(im0, (x, y), (w, h), color, 2)

        cv2.imshow('frame2', im0)
        if None in predf:
            continue
        # print(img.shape)
        # print(im0.shape)
        face_xyxy = [int(x) for x in predf[0][0:4]]
        eye1_xyxy = [int(x) for x in predf[1][0:4]]
        eye2_xyxy = [int(x) for x in predf[2][0:4]]
        # print("face_xyxy=",face_xyxy)

        # img = img.flip(dims=[3])
        # face = img[0, :, face_xyxy[0]:face_xyxy[2], face_xyxy[1]:face_xyxy[3]]
        # eye1为右眼，eye2为左眼
        face = img[:, :, face_xyxy[1]:face_xyxy[3], face_xyxy[0]:face_xyxy[2]]
        eye1 = img[:, :, eye1_xyxy[1]:eye1_xyxy[3], eye1_xyxy[0]:eye1_xyxy[2]]
        eye2 = img[:, :, eye2_xyxy[1]:eye2_xyxy[3], eye2_xyxy[0]:eye2_xyxy[2]]
        reSize = transforms.Resize([224, 224])
        face = reSize(face)
        eye1 = reSize(eye1)
        eye2 = reSize(eye2)
        print(face.shape, eye1.shape, eye2.shape)
        # face = torch.FloatTensor(face)
        # eye1 = torch.FloatTensor(eye1)
        # eye2 = torch.FloatTensor(eye2)
        # print(img[0])
        # print("faceshape=",face.shape)
        # toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        # pic1 = toPIL(img[0])
        # pic2 = toPIL(eye1)

        ylen, xlen = img[0][0].shape
        print(xlen,ylen,img[0].shape)
        xyxy = [face_xyxy[0]/xlen*25, face_xyxy[1]/ylen*25, face_xyxy[2]/xlen*25,face_xyxy[3]/ylen*25]
        print(xyxy)

        faceGrid = makeGrid(xyxy)
        faceGrid = torch.FloatTensor(faceGrid).to(device)
        faceGrid = faceGrid.reshape(1,25,25)
        # print(faceGrid.reshape(25, 25))


        gaze = tracker(face, eye2, eye1, faceGrid)
        print(gaze)
        x = gaze[0][0].item() * ppcmw
        y = gaze[0][1].item() * ppcmh

        xx = int((x + 1920 // 2))
        yy = int(y)
        print(xx, yy)
        cv2.imshow('cursor', cur)
        cv2.moveWindow("cursor", xx, -yy)

        # print(face)
        if cv2.waitKey(1) == ord('q'):
            break

    # root.mainloop()

if __name__ == '__main__':
    test2()

