import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QRect, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QMessageBox, QWidget

class gazebox(QWidget):
    def __init__(self, parent=None):
        super(gazebox, self).__init__(parent)
        # self.pix = QBitmap(img_path)  # 蒙版
        self.w = 1920
        self.h = 1080
        self.w_cnt = 4
        self.h_cnt = 3
        self.resize(self.w, self.h)
        self.rects = []
        self.gazecusor = QtWidgets.QLabel(self)
        self.gazecusor.setGeometry(QtCore.QRect(0, 0, self.w//self.w_cnt, self.h//self.h_cnt))
        self.gazecusor.setStyleSheet("background-color:rgba(221, 2, 2, 100);")
        for i in range(3):
            for j in range(4):
                self.rects.append(QRect(j*self.w//self.w_cnt,
                                        i*self.h//self.h_cnt,
                                        self.w//self.w_cnt,
                                        self.h//self.h_cnt))
        # self.setMask(self.pix)
        # self.setWindowFlags(Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)  # 设置无边框和置顶窗口样式
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # self.setWindowOpacity(0.2)

    def paintEvent(self, QPaintEvent):  # 绘制窗口
        paint = QPainter(self)
        paint.setPen(QPen(Qt.red,2,Qt.SolidLine))
        print(len(self.rects))
        for rect in self.rects:
            paint.drawRect(rect)
        # rect = QRect(0, 0, 200, 300)
        # paint.drawRect(rect)
        # paint.drawPixmap(0, 0, self.pix.width(), self.pix.height(), QPixmap(img_path))
        # self.move(10, 10)

    def moveCursor(self, pos_x, pos_y):
        height = self.h // self.h_cnt
        wide = self.w // self.w_cnt
        x = pos_x // wide
        y = pos_y // height
        self.gazecusor.move(x*wide, y*height)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my = gazebox()
    my.show()
    sys.exit(app.exec_())