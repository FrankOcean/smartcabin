from PyQt5 import QtWidgets, QtCore
from main_window import Ui_MainWindow

# 系统主菜单
class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    _signal = QtCore.pyqtSignal(str)  # 定义信号,定义参数为str类型

    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        self.derain.triggered.connect(self.testClick)
        self._signal.connect(self.mySignal)  # 将信号连接到函数mySignal

    # 打开去雨窗口
    def open_derain_window(self):
        pass

    # 打开去尘窗口
    def open_dehaze_window(self):
        pass

    # 打开眼动窗口
    def open_eyetrack_window(self):
        pass

    # 打开人脸识别窗口
    def open_facerec_window(self):
        pass

    # 打开手势识别窗口
    def open_gesturerec_window(self):
        pass

    # 打开语义分割窗口
    def open_segmentic_window(self):
        pass

    # 测试事件
    def testClick(self):
        self._signal.emit("点击事件接收")

    def mySignal(self, string):
        print(string)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())