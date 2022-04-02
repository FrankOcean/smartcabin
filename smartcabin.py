from PyQt5 import QtWidgets, QtCore
from main_window import Ui_MainWindow
from derain import Ui_Form

# 系统主菜单
class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    _signal = QtCore.pyqtSignal(str)  # 定义信号,定义参数为str类型

    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        self.derainbtn.triggered.connect(self.window_derain_show)
        self._signal.connect(self.mySignal)  # 将信号连接到函数mySignal

        self.child = ChildrenForm()

    def window_derain_show(self):
        self.gridLayout.addWidget(self.child)
        self.child.show()

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
        self.childShow()

    def mySignal(self, string):
        print(string)

# derain children window
class ChildrenForm(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(ChildrenForm, self).__init__()
        self.setupUi(self)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    myshow.show()
    sys.exit(app.exec_())