import cv2
import numpy as np
from PIL import Image


# def init(self):
#     # 获取模板样本
#     self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#
#     # 获取模板的尺寸
#     self.w = self.template.shape[0]
#     self.h = self.template.shape[1]
#
#     methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF',
#                'cv2.TM_SQDIFF_NORMED']
#     self.method = cv2.TM_CCORR

def get_match(template, method, img, width, height):
    res = cv2.matchTemplate(img, template, method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    print("----------------------------")
    print("min_val", min_val)
    print("max_val", max_val)
    print("min_loc", min_loc)
    print("max_loc", max_loc)
    print("----------------------------")
    bottom_right = (top_left[0] + width, top_left[1] + height)
    # cv2.rectangle(img, top_left, bottom_right, 255, 2)
    # cv2.circle(img, top_left, 40, (0, 0, 0), -1)  # 画圆形
    # cv2.circle(img, bottom_right, 40, (0, 0, 0), -1)  # 画圆形
    # cv2.imshow('result111', img)
    # cv2.waitKey(0)
    # c_x, c_y = ((np.array(top_left) + np.array(bottom_right)) / 2).astype(np.int)
    # print(c_x, c_y)
    return max_val, top_left, bottom_right
def testFun():
    # tmpImgName = '../template/template.png'
    # destImgName = '../img_test/dest.png'
    tmpImgName = './img_new/img00.png'
    destImgName = './img_new/img15.png'
    templateImg = cv2.imread(tmpImgName)
    destImg = cv2.imread(destImgName)
    tmpWidth = templateImg.shape[0]
    tmpHeight = templateImg.shape[1]
    destGray = cv2.cvtColor(destImg, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('imageGray', imageGray)
    # cv2.imshow('templateGray', templateGray)
    # cv2.waitKey(0)

    # 模版检测--------------------------
    maxval,t_left, b_right = get_match(templateGray, cv2.TM_CCORR, destGray, tmpWidth, tmpHeight)
    # cv2.rectangle(destGray, t_left, b_right, 255, 2)
    # cv2.imshow('result111', destGray)
    # cv2.waitKey(0)

    # 边界检测--------------------------

    # 高斯除噪
    kernel = np.ones((6, 6), np.float32) / 36
    gray_cut_filter2D = cv2.filter2D(destGray[t_left[1]:t_left[1] + tmpHeight, t_left[0]:t_left[0] + tmpWidth], -1, kernel)

    # cv2.imshow('test', gray_cut_filter2D)
    # cv2.waitKey(0)

    # 灰度图 二值化
    # gray_img = cv2.cvtColor(gray_cut_filter2D, cv2.COLOR_BGR2GRAY)
    # ret, thresh1 = cv2.threshold(gray_cut_filter2D, 180, 255, cv2.THRESH_BINARY)
    preDstImg = destGray[t_left[1]:t_left[1] + tmpHeight, t_left[0]:t_left[0] + tmpWidth]
    ret, thresh1 = cv2.threshold(preDstImg, 88, 255, cv2.THRESH_BINARY)

    # cv2.imshow('test', thresh1)
    # cv2.waitKey(0)

    # 二值化后 分割主要区域 减小干扰 模板图尺寸656*655
    tm = thresh1.copy()
    test_main = tm[100:1000, 100:1000]
    # test_main = tm[50:605, 50:606]
    # cv2.imshow('test', test_main)
    # cv2.waitKey(0)

    # 边缘化检测
    edges = cv2.Canny(test_main, 50, 150, apertureSize=3)

    # cv2.imshow('test', edges)
    # cv2.waitKey(0)

    # 霍夫直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
    if lines is None:
        print('')
    result = edges.copy()

    for line in lines[0]:
        rho = line[0]  # 第一个元素是距离rho
        theta = line[1]  # 第二个元素是角度theta
        print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
        lbael_text = 'distance:' + str(round(rho)) + 'theta:' + str(round((theta / np.pi) * 180 - 90, 2))
        cv2.putText(destGray, lbael_text, (t_left[0], t_left[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 垂直直线
            # 该直线与第一行的交点
            pt1 = (int(rho / np.cos(theta)), 0)
            # 该直线与最后一行的焦点
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            # 绘制一条白线
            cv2.line(result, pt1, pt2, 255, 1)
            # print('theat >180 theta<90')

        else:  # 水平直线
            # 该直线与第一列的交点
            pt1 = (0, int(rho / np.sin(theta)))
            # 该直线与最后一列的交点
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            # 绘制一条直线
            cv2.line(result, pt1, pt2, 255, 1)
            # print('theat <180 theta > 90')

    cv2.imshow('result', result)
    cv2.imshow('rectangle', destGray)
    cv2.waitKey(0)
    print('---------')

def preProcessImg():
    inputImg = '../img_test/testc.png'
    outputImg = '../img_test_corrected/testc.png'
    tmpImg = '../template/tmpc.png'
    method = cv2.TM_CCOEFF_NORMED
    template = cv2.imread(tmpImg)
    gray = cv2.imread(inputImg, 0)
    # res = cv2.matchTemplate(gray, template, method)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # pp1 = (0,0)
    # pp2 = (500, 500)
    # cv2.line(template, pp1, pp2, (0,255, 0), cv2.LINE_AA)
    # cv2.imshow('result111', template)
    # cv2.waitKey(0)


    # (startX, startY) = max_loc
    # endX = startX + template.shape[1]
    # endY = startY + template.shape[0]

    # 边缘化检测
    edges = cv2.Canny(template, 50, 150, apertureSize=3)
    # 霍夫直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
    if lines is None:
        print("---")
    result = edges.copy()

    # cv2.rectangle(gray, (startX, startY), (endX, endY), (255, 0, 0), 3)
    for line in lines[0]:
        rho = line[0]  # 第一个元素是距离rho
        theta = line[1]  # 第二个元素是角度theta
        print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
        lbael_text = 'distance:' + str(round(rho)) + 'theta:' + str(round((theta / np.pi) * 180 - 90, 2))
        # cv2.putText(template, lbael_text, (t_left[0], t_left[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 从图像边界画出延长直线
            # 该直线与第一行的交点
            pt1 = (int(rho / np.cos(theta)), 0)
            # 该直线与最后一行的焦点
            pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
            # 绘制一条白线
            # cv2.line(template, pt1, pt2, (0, 255, 0), cv2.LINE_AA)
            # print('theat >180 theta<90')
            cv2.circle(template, pt1, 40, (255, 255, 0), -1)  # 画圆形
            cv2.circle(template, pt2, 40, (255, 255, 0), -1)  # 画圆形
        else:  # 水平直线
            # 该直线与第一列的交点
            pt1 = (0, int(rho / np.sin(theta)))
            # 该直线与最后一列的交点
            pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
            # 绘制一条直线
            cv2.line(result, pt1, pt2, 255, 1)

    cv2.imshow('result111', template)
    cv2.waitKey(0)
    print('------')

    # cv2.rectangle(gray, t_left, b_right, 255, 2)

    # cv2.imshow('image', gray)
    # cv2.waitKey(0)
    # cv2.imwrite(outputImg, gray)

def degree2num(corrected_img_path):
    """get the class1 pointer degree and map to the number

    :param corrected_img_path: the corrected image path; eg: "./img_test_corrected/test1.png"
    :return: Instrument number
    """
    # read the image and convert to gray image
    gray = cv2.imread(corrected_img_path, 0)

    # Image edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # downsample the image for saving calculating time
    edges_img = Image.fromarray(edges)
    w, h = edges_img.size
    edges_img_resized = edges_img.resize((w // 3, h // 3))
    edges_img_resized_array = np.array(edges_img_resized)

    # use Hough Circle Transform to detect the dashboard of reduced images
    circles = cv2.HoughCircles(edges_img_resized_array, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=150, param2=100, minRadius=0, maxRadius=0)
    circles_int = np.uint16(np.around(circles))  # for visualizing
    x, y, _ = circles[0][0]  # suppose to find the biggest cycle ！！！！！！！！
    x, y = x * 3, y * 3  # map the cycle center to source image

    # detect the lines
    minLineLength = 120
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap).squeeze(1)

    """Detect the pointer line using a prior conditions: 
        1. a straight line passes through the cycle center; 
        2. the length of the line segment of the pointer is the longest
    """
    current_lines = []
    for x1, y1, x2, y2 in lines:
        # pass through the cycle center
        error = np.abs((y2 - y) * (x1 - x) - (y1 - y) * (x2 - x))
        if error < 1000:  # can change the threshold ！！！！！！
            current_lines.append((x1, y1, x2, y2))
            # for visualizing
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # find the longest line
    pointer_line = ()
    pointer_length = 0
    for x1, y1, x2, y2 in current_lines:
        length = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
        if length > pointer_length:
            pointer_length = length
            pointer_line = (x1, y1, x2, y2)

    # for visualizing
    x1, y1, x2, y2 = pointer_line
    cv2.line(gray, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # compute the pointer degree
    pointer_grad = np.abs(x2 - x1) / np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    poiner_degree = np.arccos(pointer_grad) / np.pi * 180

    # The center of the circle is compared to determine
    # the position of the pointer and then obtain the real pointer degree
    if x1 > x and y1 < y:  # In the first quadrant
        poiner_degree = poiner_degree
    elif x1 < x and y1 < y:  # In the second quadrant
        poiner_degree = 180 - poiner_degree
    elif x1 < x and y1 > y:  # In the third quadrant
        poiner_degree = 180 + poiner_degree
    else:  # In the fourth quadrant
        poiner_degree = 360 - poiner_degree

    print(poiner_degree)
    # map the degree to num
    num = 0.56  # from the map (poiner_degree to num)

    # for visualizing
    for i in circles_int[0, :]:
        # draw the outer circle
        cv2.circle(edges_img_resized_array, (i[0], i[1]), i[2], (255, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(edges_img_resized_array, (i[0], i[1]), 2, (255, 0, 0), 3)

    # show the result
    # cv2.imshow("edges", edges)
    # cv2.imshow("img", gray)
    # cv2.imshow("edges_resized", edges_img_resized_array)
    # cv2.waitKey(0)
    print(num)

    return num


if __name__ == "__main__":
    # corrected_img_path = "../img_test_corrected/test1.png"
    # degree = degree2num(corrected_img_path)
    # print(degree)
    # preProcessImg()
    # testFun()
    # queryImagePath = "../../img_test/111.png"  # the image to be corrected
    # templateImgDir = "../../template_img/pointer_meter/"  # the tamplate dir
    # outImg = "../../img_test_corrected/"
    # matchedTemplateClass = img_match.CorrectImage(queryImagePath, templateImgDir, outImg)
    print('------\n')
