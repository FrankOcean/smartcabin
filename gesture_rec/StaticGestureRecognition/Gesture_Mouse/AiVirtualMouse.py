import cv2
import gesture_rec.StaticGestureRecognition.Gesture_Mouse.HandTrackingModule as htm
import autopy #屏幕控制
import numpy as np
import time
import pyautogui


class value:
    def __init__(self) -> None:
        super().__init__()
        #矩形参数  （frameR，frameR）--（wCam - frameR, hCam - frameR）
        self.wCam, self.hCam = 650, 550
        self.frameR = 150
        self.smoothening = 2 # 鼠标移动速度
        self.i = 0# 截屏图片序号
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.pTime = 0
#返回电脑主屏幕的大小  ，以像素点（points）为单位。  1536.0 ， 864.0
        self.wScr, self.hScr = autopy.screen.size()
v = value()
detector = htm.handDetector()

def static_aivirtualMouse(img):
    img = detector.findHands(img)
    cv2.rectangle(img, (v.frameR, v.frameR), (v.wCam - v.frameR, v.hCam - v.frameR), (0, 255, 0), 2,
                  cv2.FONT_HERSHEY_PLAIN)  # 画矩形，控制区域
    lmList = detector.findPosition(img, draw=False)  # 21个关键点的真实坐标

    # 2. 判断食指和中指是否伸出
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # 食指指尖坐标
        x2, y2 = lmList[12][1:]  # 中指指尖的坐标
        fingers = detector.fingersUp()  # 五个手指全部伸出 [1, 1, 1, 1, 1] [拇指、食指、.....]
        #print(fingers)
        # 3. 若只有食指伸出 则进入移动模式
        if fingers[1] and fingers[2] == 0:
            # 4. 坐标转换： 将食指在窗口坐标转换为鼠标在桌面的坐标
            # 鼠标坐标
            # 把x1，y1在视频框中的坐标等比映射到电脑屏幕
            x3 = np.interp(x1, (v.frameR, v.wCam - v.frameR), (0, v.wScr))
            y3 = np.interp(y1, (v.frameR, v.hCam - v.frameR), (0, v.hScr))
            # smoothening values 移动速度控制
            v.clocX = v.plocX + (x3 - v.plocX) / v.smoothening
            v.clocY = v.plocY + (y3 - v.plocY) / v.smoothening
            # 鼠标移动 镜像原因  wScr - clocX
            autopy.mouse.move(v.wScr - v.clocX, v.clocY)
            cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
            v.plocX, v.plocY = v.clocX, v.clocY

        # 5. 若是食指和中指都伸出 可以移动鼠标，并且则检测指头距离 距离够短则对应鼠标点击  比ye
        if fingers[2] and fingers[1]  and fingers[0]==0 and fingers[3] == 0 and fingers[4] ==0:
            # if fingers[0] and fingers[1]==False and fingers[2] and fingers[3] and fingers[4] :
            length, img, pointInfo = detector.findDistance(8, 12, img)
            # 双指控制鼠标移动
            x3 = np.interp(pointInfo[4], (v.frameR, v.wCam - v.frameR), (0, v.wScr))
            y3 = np.interp(pointInfo[5], (v.frameR, v.hCam - v.frameR), (0, v.hScr))
            v.clocX = v.plocX + (x3 - v.plocX) / v.smoothening
            v.clocY = v.plocY + (y3 - v.plocY) / v.smoothening
            autopy.mouse.move(v.wScr - v.clocX, v.clocY)
            v.plocX, v.plocY = v.clocX, v.clocY

            # 计算距离，判断是否要长按鼠标左键
            if length < 40:
                cv2.circle(img, (pointInfo[4], pointInfo[5]), 12, (0, 255, 0), cv2.FILLED)
                img = cv2.flip(img, 1)
                # ----------鼠标点击
                # autopy.mouse.click()
                # ----------------------按下指定鼠标键
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, True)
                # pyautogui.dragTo(wScr-x3, y3, 0.1, button='left')
                return img
            else:
                autopy.mouse.toggle(autopy.mouse.Button.LEFT, False)  # 释放指定鼠标键

        # 6.截图功能  手势为  ok
        if fingers[0] and fingers[2] and fingers[3]  and fingers[4] and fingers[1] == 0:
            length, img, pointInfo = detector.screenCapture(4, 8, 12, 16, 20, img)
            if length < 40:
                # ----------截屏
                v.i += 1
                path = f'gesture_rec/cap_image/{v.i}.png'
                autopy.bitmap.capture_screen().save(path)
                cv2.putText(img,f'{v.i}.png is saved' , (15, 55),cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # 7.放大缩小
        if fingers[0] and fingers[4] and fingers[2] and fingers[3] and fingers[1]:
            pyautogui.hotkey('ctrl', '+')
        if fingers[4] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[1] == 0:
            pyautogui.hotkey('ctrl', '-')

    v.cTime = time.time()
    fps = 1 / (v.cTime - v.pTime)
    v.pTime = v.cTime
    img = cv2.flip(img, 1)
    #cv2.putText(img, f'fps:{int(fps)}', (15, 25),cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return img
