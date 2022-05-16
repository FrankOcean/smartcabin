import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPixmap, QPainter, QBitmap, QCursor
import PyQt5.QtCore as QtCore
import os
img_path = os.path.dirname(os.path.abspath(__file__)) + '/sq40.jpg'

class Cursor(QWidget):  # 不规则窗体
    def __init__(self, parent=None):
        super(Cursor, self).__init__(parent)
        self.pix = QBitmap(img_path)  # 蒙版
        self.resize(self.pix.size())
        # self.setMask(self.pix)
        self.setWindowFlags(Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)  # 设置无边框和置顶窗口样式

    def paintEvent(self, QPaintEvent):  # 绘制窗口
        paint = QPainter(self)
        paint.drawPixmap(0, 0, self.pix.width(), self.pix.height(), QPixmap(img_path))
        self.move(10, 10)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    t = Cursor()
    t.show()
    sys.exit(app.exec_())
