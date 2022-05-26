import sys
import numpy as np
import cv2
import torch

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox, QWidget

import eye_track.pytorch.YoloItracker
from eye_track.cursor import Cursor
from eye_track.pytorch.YoloItracker import pred_once, process_img
from eye_track.remove_glass.MyRemoval import remove_glass
from eyetrack import Ui_EyetrackForm

class SmartEyetrackWindow(QWidget, Ui_EyetrackForm):

    def __init__(self):

        super(SmartEyetrackWindow, self).__init__()
        self.resize(967, 688)

        self.btn_show_cursor = QtWidgets.QPushButton(self)
        self.btn_show_cursor.setGeometry(QtCore.QRect(490, 590, 90, 30))
        self.btn_show_cursor.setObjectName("btn_show_cursor")
        self.btn_stop = QtWidgets.QPushButton(self)
        self.btn_stop.setGeometry(QtCore.QRect(690, 590, 90, 30))
        self.btn_stop.setObjectName("stop")
        self.btn_start = QtWidgets.QPushButton(self)
        self.btn_start.setGeometry(QtCore.QRect(90, 590, 90, 30))
        self.btn_start.setObjectName("btn_start")
        self.btn_remove_glass = QtWidgets.QPushButton(self)
        self.btn_remove_glass.setGeometry(QtCore.QRect(290, 590, 90, 30))
        self.btn_remove_glass.setObjectName("btn_remove_glass")
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(120, 90, 640, 480))
        self.label.setStyleSheet("background-color:rgb(221, 255, 194);")
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.btn_close = QtWidgets.QPushButton(self)
        self.btn_close.setGeometry(QtCore.QRect(690, 30, 90, 30))
        self.btn_close.setObjectName("btn_close")

        self.retranslateUi(self)
        self.btn_close.clicked.connect(self.close)
        QtCore.QMetaObject.connectSlotsByName(self)

        # my code
        self.fps = 33  # 33频率 合计30fps
        self.frame = []  # 存图片
        self.detectFlag = True  # 检测flag

        self.cap = []
        self.timer_camera = QTimer()  # 定义定时器
        self.removeGlassFlag = False
        self.cursor = Cursor()
        self.cursor.show()
        self.showCursorFlag = False
        self.cursor.setVisible(self.showCursorFlag)
        # connect slot
        self.btn_start.clicked.connect(self.slotStart)
        self.btn_stop.clicked.connect(self.slotStop)
        self.btn_show_cursor.clicked.connect(self.slotShowCursor)
        self.btn_remove_glass.clicked.connect(self.slotRemoveGlass)


    def retranslateUi(self, EyetrackForm):
        _translate = QtCore.QCoreApplication.translate
        EyetrackForm.setWindowTitle(_translate("EyetrackForm", "Form"))
        self.btn_show_cursor.setText(_translate("EyetrackForm", "显示注视点"))
        self.btn_stop.setText(_translate("EyetrackForm", "停止"))
        self.btn_start.setText(_translate("EyetrackForm", "开始"))
        self.btn_remove_glass.setText(_translate("EyetrackForm", "去除眼镜"))
        self.label.setText(_translate("EyetrackForm",
                                      "<html><head/><body><p><span style=\" font-style:italic;\">等待视频输入</span></p></body></html>"))
        self.btn_close.setText(_translate("EyetrackForm", "关闭"))

    def slotStart(self):
        """ Slot function to start the progamme
            """

        # videoName, _ = QFileDialog.getOpenFileName(self, "Open", "", "*.avi;;*.mp4;;All Files(*)")
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
        self.cursor.setVisible(True)
        self.timer_camera.start(self.fps)
        self.timer_camera.timeout.connect(self.openFrame)

    def slotShowCursor(self):
        print("show cursor")
        self.showCursorFlag = not self.showCursorFlag
        if self.showCursorFlag == True:
            self.btn_show_cursor.setText("隐藏注视点")
        else:
            self.btn_show_cursor.setText("显示注视点")

        self.cursor.setVisible(self.showCursorFlag)

    def slotRemoveGlass(self):
        print("remove glass")
        self.removeGlassFlag = not self.removeGlassFlag

        if self.removeGlassFlag == True:
            self.btn_remove_glass.setText("去除眼镜")
        else:
            self.btn_remove_glass.setText("显示眼镜")

    def slotStop(self):
        """ Slot function to stop the programme
            """
        self.cursor.setVisible(False)
        if self.cap != []:

            self.cap.release()
            self.timer_camera.stop()  # 停止计时器
            self.label.setText("This video has been stopped.")
            self.label.setText("等待视频输入")
            self.label.setStyleSheet("background-color:rgb(221, 255, 194);font-style:italic;")
        else:
            self.label_num.setText("Push the left upper corner button to Quit.")
            Warming = QMessageBox.warning(self, "Warming", "Push the left upper corner button to Quit.",
                                          QMessageBox.Yes)

    def openFrame(self):
        """ Slot function to capture frame and process it
            """
        if (self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret:
                xx = yy = 0
                if self.removeGlassFlag == True:
                    frame = remove_glass(frame)
                # frame = frame.resize(640, 480)
                frame = np.expand_dims(frame, 0)
                # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                if self.detectFlag == True:
                    # 检测代码self.frame
                    img, im0 = process_img(frame)
                    lst = pred_once(img, im0)
                    if len(lst) == 3:
                        [xx, yy, frame2] = lst
                    else:
                        frame2 = lst[-1]
                self.cursor.move(xx, yy)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame2.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame2.data, width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(QPixmap.fromImage(q_image))
            else:
                self.cap.release()
                self.timer_camera.stop()  # 停止计时器



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = SmartEyetrackWindow()
    my.show()
    sys.exit(app.exec_())