import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.6, trackCon=0.6):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # 导入solution
        self.mpHands = mp.solutions.hands
        # 导入模型
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,  # 是静态图片还是连续视频帧
                                        max_num_hands=self.maxHands, # 最多检测几只手
                                        min_detection_confidence=self.detectionCon, # 置信度阈值
                                        min_tracking_confidence=self.trackCon) # 追踪阈值
        # 导入绘图函数
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 检测手
        self.results = self.hands.process(imgRGB)

        # print(self.results.multi_handedness)  # 获取检测结果中的左右手标签、和置信度
        # print(self.results.multi_handedness[0].classification[0].label)
        # print(self.results.multi_hand_landmarks) # 左右手 列表

        # 如果有检测到手
        if self.results.multi_hand_landmarks:
            if self.results.multi_handedness[0].classification[0].label == "Right":
                return True
            else:
                return False

