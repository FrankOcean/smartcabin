# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1032, 758)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1032, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        self.menu_6 = QtWidgets.QMenu(self.menubar)
        self.menu_6.setObjectName("menu_6")
        self.menu_7 = QtWidgets.QMenu(self.menubar)
        self.menu_7.setObjectName("menu_7")
        MainWindow.setMenuBar(self.menubar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionderain = QtWidgets.QAction(MainWindow)
        self.actionderain.setObjectName("actionderain")
        self.actiondehaze = QtWidgets.QAction(MainWindow)
        self.actiondehaze.setObjectName("actiondehaze")
        self.actioneyetrack = QtWidgets.QAction(MainWindow)
        self.actioneyetrack.setObjectName("actioneyetrack")
        self.actionfacerec = QtWidgets.QAction(MainWindow)
        self.actionfacerec.setObjectName("actionfacerec")
        self.actiongesture = QtWidgets.QAction(MainWindow)
        self.actiongesture.setObjectName("actiongesture")
        self.actionimglink = QtWidgets.QAction(MainWindow)
        self.actionimglink.setObjectName("actionimglink")
        self.actionsemantic = QtWidgets.QAction(MainWindow)
        self.actionsemantic.setObjectName("actionsemantic")
        self.menu.addAction(self.actioneyetrack)
        self.menu_2.addAction(self.actionfacerec)
        self.menu_3.addAction(self.actiongesture)
        self.menu_4.addAction(self.actionderain)
        self.menu_5.addAction(self.actiondehaze)
        self.menu_6.addAction(self.actionimglink)
        self.menu_7.addAction(self.actionsemantic)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())
        self.menubar.addAction(self.menu_6.menuAction())
        self.menubar.addAction(self.menu_7.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "smart_v0"))
        self.menu.setTitle(_translate("MainWindow", "眼动追踪"))
        self.menu_2.setTitle(_translate("MainWindow", "人脸识别"))
        self.menu_3.setTitle(_translate("MainWindow", "手势识别"))
        self.menu_4.setTitle(_translate("MainWindow", "视频去雨"))
        self.menu_5.setTitle(_translate("MainWindow", "视频去尘"))
        self.menu_6.setTitle(_translate("MainWindow", "全景拼接"))
        self.menu_7.setTitle(_translate("MainWindow", "语义分割"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionderain.setText(_translate("MainWindow", "derain"))
        self.actiondehaze.setText(_translate("MainWindow", "dehaze"))
        self.actioneyetrack.setText(_translate("MainWindow", "eyetrack"))
        self.actionfacerec.setText(_translate("MainWindow", "facerec"))
        self.actiongesture.setText(_translate("MainWindow", "gesture"))
        self.actionimglink.setText(_translate("MainWindow", "imglink"))
        self.actionsemantic.setText(_translate("MainWindow", "semantic"))
