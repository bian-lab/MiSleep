# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\MiSleep\gui\load_data\detail.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_detail(object):
    def setupUi(self, detail):
        detail.setObjectName("detail")
        detail.resize(630, 412)
        self.verticalLayout = QtWidgets.QVBoxLayout(detail)
        self.verticalLayout.setObjectName("verticalLayout")
        self.detailEdit = QtWidgets.QPlainTextEdit(detail)
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei UI")
        font.setPointSize(10)
        self.detailEdit.setFont(font)
        self.detailEdit.setObjectName("detailEdit")
        self.verticalLayout.addWidget(self.detailEdit)
        self.confirmBt = QtWidgets.QPushButton(detail)
        self.confirmBt.setMinimumSize(QtCore.QSize(0, 40))
        font = QtGui.QFont()
        font.setFamily("Microsoft JhengHei UI")
        font.setPointSize(12)
        self.confirmBt.setFont(font)
        self.confirmBt.setObjectName("confirmBt")
        self.verticalLayout.addWidget(self.confirmBt)

        self.retranslateUi(detail)
        QtCore.QMetaObject.connectSlotsByName(detail)

    def retranslateUi(self, detail):
        _translate = QtCore.QCoreApplication.translate
        detail.setWindowTitle(_translate("detail", "Detail"))
        self.detailEdit.setPlainText(_translate("detail", "original file: \n"
"animal number:\n"
"date of experiment:\n"
"number of experiment:\n"
"start time:\n"
"number of channels:\n"
"sampling rate:\n"
"Neurologger:\n"
"type of battery:\n"
"other setup:\n"
"comments:\n"
"channel1:\n"
"channel2:"))
        self.confirmBt.setText(_translate("detail", "Confirm and MiSleep"))
