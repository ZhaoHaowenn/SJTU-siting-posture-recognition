import numpy as np
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def getLength (pt1,pt2):
    distance = math.pow((pt1[0] - pt2[0]),2) + math.pow((pt1[1] - pt2[1]),2)
    distance = math.sqrt(distance)
    return distance

def getAngle(pt1,pt2,pt3):
    a = getLength(pt2, pt3)
    b = getLength(pt1, pt2)
    c = getLength(pt1, pt3)
    A = math.acos((b*b + c*c - a*a) / (2 * b*c)) * 180 / 3.1415;
    return A

# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

# draw the body keypoint and lims

def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4            #线条宽度
    #肢体部位连接的序列
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    #定义颜色,用于绘制不同肢体部位的连接线和关节点
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    c = input("请输入摄像头角度（左正右负：-90 ~ 90）：")
    c1 = int(c)
    d = input("是否进行单独输出：")
    for n in range(len(subset)):
        test = ""
        points = []
        for i in range(18):
            index = int(subset[n][i])
            if index == -1:
                points.append(None)
                continue
            x, y = candidate[index][0:2]
            points.append((int(x),int(y)))
            cv2.circle(canvas, (points[i][0], points[i][1]), 4, colors[n%18], thickness=-1)

        for i in range(17):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[n%18])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        if points[0] :
            canvas = cv2AddChineseText(canvas, "目标" + str(n + 1), (points[0][0], points[0][1]), (255, 0, 0), 20)
        if c1 < 60 and c1 > -60:
            if points[1] and points[0] and points[2]:
                    A1 = getAngle(points[1], points[0], points[2])  # A1为鼻子、脖子和右肩的夹角
                    if A1 < 75 - c1/1.5:
                        test = test + " 头部右倾"
            if points[1] and points[0] and points[5]:
                    A2 = getAngle(points[1], points[0], points[5])  # A2为鼻子、脖子和左肩的夹角
                    if A2 < 75 + c1/1.5:
                        test = test + " 头部左倾"
            if points[1] and points[2] and points[5]:
                if points[2][1] > (points[1][1] + 0.12*(1 + abs(c1)/45)*getLength(points[2],points[5])) and points[1][1] > (points [5][1] + 0.12*(1 + abs(c1)/45)*getLength(points[2],points[5])):
                    test = test + " 身体右倾"
                if points[2][1] < (points[1][1] - 0.12*(1 + abs(c1)/45)*getLength(points[2],points[5])) and points[1][1] < (points [5][1] - 0.12*(1 + abs(c1)/45)*getLength(points[2],points[5])):
                    test = test + " 身体左倾"

            if  ("身体左倾" in test or "身体右倾" in test or "头部左倾" in test or "头部右倾" in test) and (-30 < c1 < 30):
                  pass
            else:
                if points[0]:
                    if 0 <= c1 < 60 and points[1] and points[2]:
                        a = points[1][1] - points[0][1]  # a为鼻子和脖子之间垂直的距离
                        b = getLength(points[2], points[1])  # b为脖子和右肩的距离
                        if a < b * (1 + abs(c1)/45)*0.5 :
                            test = test + " 低头"
                    elif 0 > c1 > -60 and points[1] and points[5]:
                        a = points[1][1] - points[0][1]  # a为鼻子和脖子之间垂直的距离
                        b = getLength(points[5], points[1])  # b为左肩和脖子的距离
                        if a < b * (1 + abs(c1)/45)*0.5:
                            test = test + " 低头"
                else:
                    test = test + " 低头"
        if  c1 >= 60 or c1 <= -60:
            if c1 < -60 and points[1] and points[8] :
                A4 = getAngle(points[8],points[1],(points[8][0] - c1,points[8][1]))
                if A4 < 65:
                    test = test + " 身体前倾"
                if A4 > 125:
                    test = test + " 身体后仰"
            if c1 > 60 and points[1] and points[11] :
                A4 = getAngle(points[11],points[1],(points[11][0] - c1,points[11][1]))
                if A4 < 70:
                    test = test + " 身体前倾"
                if A4 > 110:
                    test = test + " 身体后仰"
            if points[1] and points[2] and points[5] and points[0]:
                if (c1 < -60) and points[2][1] > (points[1][1] + 0.2*(2-abs(c1)/90)*getLength(points[1],points[0])) and points[1][1] > (points [5][1] + 0.2*(2-abs(c1)/90)*getLength(points[1],points[0])):
                    test = test + " 身体右倾"
                if (c1 > 60) and points[2][1] < (points[1][1] - 0.2*(2-abs(c1)/90)*getLength(points[1],points[0])) and points[1][1] < (points [5][1] - 0.2*(2-abs(c1)/90)*getLength(points[1],points[0])):
                    test = test + " 身体左倾"
            if  "身体前倾" in test or "身体后仰" in test or "身体左倾" in test or "身体右倾" in test:
                pass
            else:
                if points[0] and points[1]:
                    A3 = getAngle(points[1],points[0],(points[1][0] - c1,points[1][1]))
                    if A3 < 90 - abs(c1)*0.65  :
                        test = test + " 低头"
                    if A3 > 120 - abs(c1)*0.65 :
                        test = test + " 仰头"
        if "低头" in test :
            pass
        else:
            if points[4] and points[0] and points[1]:
                if points[4][1] < points[1][1]:
                    test = test + " 举右手"
            if points[7] and points[0] and points[1]:
                if points[7][1] < points[1][1]:
                    test = test + " 举左手"
        if test == "":
            test = test + " 正常"
        print(test)
        canvas = cv2AddChineseText(canvas, "目标" + str(n + 1) + "坐姿：" + test, (10, (n - 1) * 30 + 35), (255, 0, 0),30)

        if d == "是":
            if len(subset) > 1 :
                max_x, max_y, min_x, min_y = search(points)
                image = canvas[min_y - 50:max_y + 50,min_x :max_x ]
                image = cv2AddChineseText(image,test, (0 , 20), (255, 0, 0), 25)
                cv2.imshow('estimated_image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas

# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j

def search(array):
    max_x = 0
    max_y = 0
    min_x = float('inf')
    min_y = float('inf')
    for i in range(18):
        if array[i]:
            x = array[i][0]
            y = array[i][1]
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
    return max_x,max_y,min_x,min_y
