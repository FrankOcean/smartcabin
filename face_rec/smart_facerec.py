# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from facerec import Ui_FacerecForm
import face_recognition
import os, cv2

class SmartFaceWindow(QtWidgets.QWidget, Ui_FacerecForm):

    def __init__(self):
        super(SmartFaceWindow, self).__init__()
        self.setupUi(self)
        self.load_default_settings()

    def load_default_settings(self):
        # my code
        self.fps = 13  # 13频率 合计10fps
        self.frame = []  # 存图片

        self.cap = []
        self.timer_camera = QTimer()  # 定义定时器
        # connect slot
        self.btn_start.clicked.connect(self.slotStart)
        self.btn_stop.clicked.connect(self.slotStop)

        # 读取到数据库中的人名和面部特征
        self.face_databases_dir = 'face_databases'
        self.user_names = []
        self.user_faces_encodings = []
        # 得到face_databases中所有文件
        self.files = os.listdir('face_databases')

        # 循环读取
        for image_shot_name in self.files:
            # 截取文件名作为用户名 存入user_names列表中
            user_name, _ = os.path.splitext(image_shot_name)
            self.user_names.append(user_name)
            # 读取图片文件中的面部特征信息存入user_faces_encodings列表中
            image_file_name = os.path.join(self.face_databases_dir, image_shot_name)
            # 加载图片
            image_file = face_recognition.load_image_file(image_file_name)
            # 读取图片信息
            face_encoding = face_recognition.face_encodings(image_file)[0]
        self.user_faces_encodings.append(face_encoding)

    def slotStart(self):
        """ Slot function to start the progamme
            """
        self.cap = cv2.VideoCapture(0)
        self.timer_camera.start(self.fps)
        self.timer_camera.timeout.connect(self.openFrame)

    def slotStop(self):
        """ Slot function to stop the programme
            """
        if self.cap != []:

            self.cap.release()
            self.timer_camera.stop()  # 停止计时器
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
                # 2.2从拍摄到的画面中提取出人的脸部所在区域
                face_locations = face_recognition.face_locations(frame)
                # 2.2.1从所有人的头像所在区域提取出脸部特征
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                names = []
                # 2.2.2 匹配 遍历face_encodings和数据库中的去比对
                for face_encoding in face_encodings:
                    matchers = face_recognition.compare_faces(self.user_faces_encodings, face_encoding)
                    name = "Unknown"
                    for index, is_match in enumerate(matchers):
                        if is_match:
                            name = self.user_names[index]
                            break
                    names.append(name)
                # 2.3循环遍历人到额脸部所在区域 并画框 在框框上标识姓名
                for (top, right, bottom, left), name in zip(face_locations, names):
                    # 2.3.1画框
                    # BGR
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left, top - 10), font, 0.5, (0, 255, 0), 1)
                # 2.4通过opencv把画面展示出来
                # cv2.imshow("Video", frame)

                frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame2.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame2.data, width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.label.width(), self.label.height())
                self.label.setPixmap(QPixmap.fromImage(q_image))
            else:
                cv2.destroyAllWindows()
                self.cap.release()
                self.timer_camera.stop()  # 停止计时器

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    smart = SmartFaceWindow()
    smart.show()
    sys.exit(app.exec_())