
import time
import nnTest
from sys import argv,exit
from PyQt5 import QtCore, QtWidgets
from  cv2  import threshold,cvtColor,resize,imread,INTER_LANCZOS4,THRESH_OTSU,COLOR_RGB2GRAY
from tensorflow.examples.tutorials.mnist.input_data import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class MyWindow(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super(MyWindow, self).__init__(parent)
        self.initUi()

    def initUi(self):

        self.setWindowTitle('不太准确的手写数字识别程序')

        self.resize(560, 380)
        self.setWindowIcon(QIcon('Logo.ico'))

        self.paintBoard = PaintBoard(self)
        self.paintBoard.setGeometry(QtCore.QRect(90, 90, 28 * 6, 28 * 6))

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(90, 40, 150, 40))
        self.label.setText("  请在下方书写数字：")

        self.mybutton1=QPushButton('识别',self)
        self.mybutton1.clicked.connect(self.Recognize)
        self.mybutton1.setGeometry(QtCore.QRect(390, 50, 75, 23))

        self.mybutton2 = QPushButton('清除',self)
        self.mybutton2.clicked.connect(self.paintBoard.Clear)
        self.mybutton2.setGeometry(QtCore.QRect(390, 100, 75, 23))


        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setAlignment(Qt.AlignCenter)
        self.textBrowser.setGeometry(QtCore.QRect(385, 190, 90, 138))
        self.textBrowser.setFont(QFont("仿宋", 84, QFont.Bold))

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(400, 160, 54, 12))
        self.label.setText("预测结果：")

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(90, 330, 54, 12))
        self.label_2.setText("总耗时：")

        self.textBrowser_2 = QtWidgets.QTextBrowser(self)
        self.textBrowser_2.setGeometry(QtCore.QRect(140, 325, 60, 25))




    def Recognize(self):
        savePath = "./example.png"
        image = self.paintBoard.GetContentAsQImage()
        image.save(savePath)
        time_start = time.time()
        img = imread('./example.png', 1)
        img_resize = resize(img,(int(28), int(28)), interpolation=INTER_LANCZOS4)
        # 将图片转为灰度
        img_gray = cvtColor(img_resize, COLOR_RGB2GRAY)
        # 将图片转为二值图
        ret, img_gray = threshold(img_gray, 0, 255, THRESH_OTSU)
        img_gray = img_gray.reshape(784)
        # 归一化
        img_gray = [(255 - x) * 1.0 / 255.0 for x in img_gray]
        tf.reset_default_graph()
        # 返回预测结果
        result = int(nnTest.testPic(img_gray))
        time_end = time.time()
        time_cost= time_end - time_start
        self.textBrowser.setText(str(result))
        self.textBrowser_2.setText(str(int(time_cost*1000))+"ms")


class PaintBoard(QWidget):

    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.InitData()  # 先初始化数据，再初始化界面
        self.InitView()

    def InitData(self):
        self.size = QSize(28 * 6, 28 * 6)

        # 新建QPixmap作为画板，尺寸为__size
        self.board = QPixmap(self.size)
        self.board.fill(Qt.white)  # 用白色填充画板
        self.IsEmpty = True  # 默认为空画板

        self.lastPos = QPoint(0, 0)  # 上一次鼠标位置
        self.currentPos = QPoint(0, 0)  # 当前的鼠标位置

        self.painter = QPainter()  # 新建绘图工具

        self.thickness = 7  # 默认画笔粗细为7px
        self.penColor = QColor("black")  # 设置默认画笔颜色为黑色

    def InitView(self):
        # 设置界面的尺寸为__size
        self.setFixedSize(self.size)

    def Clear(self):
        # 清空画板
        self.board.fill(Qt.white)
        self.update()
        self.IsEmpty = True

    def IsEmpty(self):
        # 返回画板是否为空
        return self.IsEmpty

    def GetContentAsQImage(self):
        # 获取画板内容（返回QImage）
        image = self.board.toImage()
        return image

    def paintEvent(self, paintEvent):
        # 绘图事件
        # 绘图时必须使用QPainter的实例，此处为painter
        # 绘图在begin()函数与end()函数间进行
        # begin(param)的参数要指定绘图设备，即把图画在哪里
        # drawPixmap用于绘制QPixmap类型的对象
        self.painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，board即要绘制的图
        self.painter.drawPixmap(0, 0, self.board)
        self.painter.end()

    def mousePressEvent(self, mouseEvent):
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.currentPos = mouseEvent.pos()
        self.lastPos = self.currentPos

    def mouseMoveEvent(self, mouseEvent):
        # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.currentPos = mouseEvent.pos()
        self.painter.begin(self.board)

        self.painter.setPen(QPen(self.penColor, self.thickness))  # 设置画笔颜色，粗细

        # 画线
        self.painter.drawLine(self.lastPos, self.currentPos)
        self.painter.end()
        self.lastPos = self.currentPos

        self.update()  # 更新显示

    def mouseReleaseEvent(self, mouseEvent):
        self.IsEmpty = False  # 画板不再为空


if __name__ == '__main__':
    app=QApplication(argv)
    myshow=MyWindow()
    myshow.setObjectName("myshow")
    myshow.setStyleSheet("#myshow{background-color: silver}")
    myshow.show()
    exit(app.exec_())


