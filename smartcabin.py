from PyQt5 import QtWidgets
from mainwindow.main_window import Ui_MainWindow
from derain.derain import Ui_DerainForm
from dehaze.dehaze import Ui_DehazeForm
from eye_track.eyetrack import Ui_EyetrackForm
from face_rec.facerec import Ui_FacerecForm
from gesture_rec.gesture import Ui_GestureForm
from semantic_seg.semantic import Ui_SemanticForm
from imglink.imglink import Ui_ImglinkForm

# 系统主菜单
class SmartWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(SmartWindow, self).__init__()
        self.setupUi(self)

        self.actionderain.triggered.connect(self.open_derain_window)
        self.actiondehaze.triggered.connect(self.open_dehaze_window)
        self.actionfacerec.triggered.connect(self.open_facerec_window)
        self.actionimglink.triggered.connect(self.open_imglink_window)
        self.actiongesture.triggered.connect(self.open_gesturerec_window)
        self.actioneyetrack.triggered.connect(self.open_eyetrack_window)
        self.actionsemantic.triggered.connect(self.open_segmentic_window)

    # 打开去雨窗口
    def open_derain_window(self):
        self.child = ChildrenDerainForm()
        self.verticalLayout.addWidget(self.child)
        self.child.show()

    # 打开去尘窗口
    def open_dehaze_window(self):
        self.child = ChildrenDehazeForm()
        self.verticalLayout.addWidget(self.child)
        self.child.show()

    # 打开眼动窗口
    def open_eyetrack_window(self):
        self.child = ChildrenEyetrackForm()
        self.verticalLayout.addWidget(self.child)
        self.child.show()

    # 打开人脸识别窗口
    def open_facerec_window(self):
        self.child = ChildrenFacerecForm()
        self.verticalLayout.addWidget(self.child)
        self.child.show()

    # 打开手势识别窗口
    def open_gesturerec_window(self):
        self.child = ChildrenGestureForm()
        self.verticalLayout.addWidget(self.child)
        self.child.show()

    # 打开语义分割窗口
    def open_segmentic_window(self):
        self.child = ChildrenSemanticForm()
        self.verticalLayout.addWidget(self.child)
        self.child.show()

    # 打开语义分割窗口
    def open_imglink_window(self):
        self.child = ChildrenImglinkForm()
        self.verticalLayout.addWidget(self.child)
        self.child.show()

    # 测试事件
    def testClick(self):
        self.msg()

    def msg(self):
        reply = QtWidgets.QMessageBox.information(self,  # 使用infomation信息框
                                        "标题",
                                        "测试用",
                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

# 1. derain children window
class ChildrenDerainForm(QtWidgets.QWidget, Ui_DerainForm):
    def __init__(self):
        super(ChildrenDerainForm, self).__init__()
        self.setupUi(self)

# 2. dehaze children window
class ChildrenDehazeForm(QtWidgets.QWidget, Ui_DehazeForm):
    def __init__(self):
        super(ChildrenDehazeForm, self).__init__()
        self.setupUi(self)

# 3. Eyetrack children window
class ChildrenEyetrackForm(QtWidgets.QWidget, Ui_EyetrackForm):
    def __init__(self):
        super(ChildrenEyetrackForm, self).__init__()
        self.setupUi(self)

# 4. gesture recognition children window
class ChildrenGestureForm(QtWidgets.QWidget, Ui_GestureForm):
    def __init__(self):
        super(ChildrenGestureForm, self).__init__()
        self.setupUi(self)

# 5. semantic segmentition window
class ChildrenSemanticForm(QtWidgets.QWidget, Ui_SemanticForm):
    def __init__(self):
        super(ChildrenSemanticForm, self).__init__()
        self.setupUi(self)

# 6. face recognition children window
class ChildrenFacerecForm(QtWidgets.QWidget, Ui_FacerecForm):
    def __init__(self):
        super(ChildrenFacerecForm, self).__init__()
        self.setupUi(self)

# 7. image link children window
class ChildrenImglinkForm(QtWidgets.QWidget, Ui_ImglinkForm):
    def __init__(self):
        super(ChildrenImglinkForm, self).__init__()
        self.setupUi(self)

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    smart = SmartWindow()
    smart.show()
    sys.exit(app.exec_())