from cv2 import EVENT_LBUTTONDOWN, EVENT_MOUSEMOVE, circle, EVENT_LBUTTONUP,namedWindow,setMouseCallback,waitKey,imshow,\
    cvtColor,resize,INTER_LANCZOS4,COLOR_RGB2GRAY,moveWindow,destroyAllWindows,\
    threshold,THRESH_OTSU,destroyAllWindows,INTER_NEAREST,imread
from numpy import zeros,uint8

drawing =False  # 鼠标左键按下时，该值为True，标记正在绘画
mode =  False# False 画圆
ix, iy = -1, -1 # 鼠标左键按下时的坐标
img = zeros((28*10, 28*10, 3), uint8)
img.fill(255)


# 设置鼠标事件的回调函数

def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == EVENT_LBUTTONDOWN:
        # 鼠标左键按下事件
        drawing = True
        ix, iy = x, y

    elif event == EVENT_MOUSEMOVE:
        # 鼠标移动事件
        if drawing == True:
            circle(img, (x, y), 7, (0, 0, 0), -1)

    elif event == EVENT_LBUTTONUP:
        # 鼠标左键松开事件
        drawing = False
        circle(img, (x, y), 7, (0, 0, 0), -1)

#   加载窗口绘制数字 返回图像矩阵
def loadPic():
    namedWindow('image')
    setMouseCallback('image', draw_circle)
    while(1):
        imshow('image', img)
        k = waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    img_resize = resize(img,
    (int(28),int(28),),interpolation=INTER_LANCZOS4)
    moveWindow("img",1000,1000)
    #将图片转为灰度图
    img_gray1 = cvtColor(img_resize,COLOR_RGB2GRAY)
    #将图片转为二值图
    ret, img_gray = threshold(img_gray1, 150, 255, THRESH_OTSU)
    #ret, img_gray = cv2.threshold(img_gray1, 10, 200, cv2.THRESH_BINARY)
    print(img_gray.shape)
    #cv2.imwrite('messigray1.png',img_gray)
    destroyAllWindows()
    img_gray = img_gray.reshape(784)
    img_gray = [(255 - x) * 1.0 / 255.0 for x in img_gray]
    return img_gray

#   加载路径下图片 返回二值图数组
def readPic():
    img = imread('example.png', 1)
    img_resize = resize(img,
    (int(28),int(28)),interpolation=INTER_NEAREST)
    img_gray = cvtColor(img_resize, COLOR_RGB2GRAY)
    # 将图片转为二值图
    # ret, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    img_gray = img_gray.reshape(784)
    img_gray = [(255 - x) * 1.0 / 255.0 for x in img_gray]
    return img_gray