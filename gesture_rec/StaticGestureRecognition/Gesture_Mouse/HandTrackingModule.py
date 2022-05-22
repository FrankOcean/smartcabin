import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.8, trackCon=0.8):
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
        # 手指指尖序号
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #检测手
        self.results = self.hands.process(img)

        #print(self.results.multi_handedness[0].classification[0].label)  # 获取检测结果中的左右手标签、和置信度
        #print(self.results.multi_hand_landmarks) # 左右手 列表

        # 如果有检测到手
        if self.results.multi_hand_landmarks:

            for hand21 in self.results.multi_hand_landmarks:
                if draw:
                    # 可视化关键点及骨架连线
                    self.mpDraw.draw_landmarks(img, hand21, self.mpHands.HAND_CONNECTIONS)
        return img

    #手部关键点坐标
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h) #关键点的真实坐标
                    # print(id, cx, cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
        return self.lmList
    #检测手指是否伸出来
    def fingersUp(self):
        fingers = []
        # 右手大拇指
        # self.tipIds = [4, 8, 12, 16, 20]
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 2][1]: #实际中大拇指指尖cx的坐标大于关键点2的坐标
            fingers.append(1)# 伸出来
        else:
            fingers.append(0)# 收回去
        # 其余手指
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] > self.lmList[self.tipIds[id] - 2][2]:#实际中出大拇指外的指尖cy的坐标小于下一位的坐标
                fingers.append(0)# 收回去
            else:
                fingers.append(1)# 伸出来
        return fingers

    # 食指与中指的距离 8  12
    def findDistance(self, p1, p2, img, draw=True, r=12, t=3):
        x1, y1 = self.lmList[p1][1:]  #获取某个关键点的 x y 坐标 8
        x2, y2 = self.lmList[p2][1:]  # 12
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  #两根手指的中点坐标
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t) #链接两个手指尖的线
            cv2.circle(img, (x1, y1), r, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)# 画圈
            length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
    #截屏
    def screenCapture(self, p1, p2, p3,p4,p5,img, draw=True, r=10, t=3):
        x1, y1 = self.lmList[p1][1:]  # 获取某个关键点的 x y 坐标 4
        x2, y2 = self.lmList[p2][1:]  # 8
        x3, y3 = self.lmList[p3][1:]  # 12
        x4, y4 = self.lmList[p4][1:]  # 16
        x5, y5 = self.lmList[p5][1:]  # 20
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 两根手指的中点坐标
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)  # 链接两个手指尖的线
            cv2.circle(img, (x1, y1), r, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x4, y4), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x5, y5), 8, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)  # 画圈
            length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        # 检测手势并画上骨架信息
        img = detector.findHands(img)
        # 获取得到坐标点的列表
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            #print(lmList) #关键点坐标列表
            #print(lmList[4]) # 大拇指坐标
            #length, img, l = detector.findDistance(4, 8, img) # 食指与中指距离判定
            fingers=detector.fingersUp() # 五指伸合结构输出
            print(fingers)
            #length, img, pointInfo = detector.screenCapture(4, 8, 12, 16, 20, img) # 截屏动作执行

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, 'fps:' + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)
        if cv2.waitKey(30) & 0xFF==27:   #esc退出
            break

if __name__ == "__main__":
    main()
