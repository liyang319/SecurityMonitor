import cv2
import numpy as np
from PIL import Image


def get_match(template, method, img, width, height):
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 设置匹配的阈值，可以根据实际情况调整
    threshold = 0.35
    # 使用np.where找到匹配结果大于阈值的位置
    locations = np.where(res >= threshold)
    # locations = list(zip(*locations[::-1]))
    # 循环遍历每个匹配结果的位置
    i = 0
    for (x, y) in zip(*locations[::-1]):
        # 在目标图片上绘制矩形框标记匹配区域
        print(str(i) + '----------(' + str(x) + ',' + str(y) + ')-------(' + str(x + width) + ',' + str(y + height) + ')----')
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        i += 1

    # 显示标记了匹配区域的目标图片
    # detectRect(img, width, height)
    cv2.imshow('Matched Objects', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    print("----------------------------")
    print("min_val", min_val)
    print("max_val", max_val)
    print("min_loc", min_loc)
    print("max_loc", max_loc)
    print("----------------------------")
    bottom_right = (top_left[0] + width, top_left[1] + height)
    return max_val, top_left, bottom_right


def getMeterResult(dstImg, width, height, wOffset, hOffset):
    retVal = ''
    ret, thresh1 = cv2.threshold(dstImg, 88, 255, cv2.THRESH_BINARY)
    # 二值化后 分割主要区域 减小干扰 模板图尺寸656*655
    tm = thresh1.copy()
    test_main = tm[hOffset:height, wOffset:width]

    # 边缘化检测
    edges = cv2.Canny(test_main, 50, 150, apertureSize=3)

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
        # cv2.putText(dstImg, lbael_text, (t_left[0], t_left[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
    cv2.imshow('result111', result)
    cv2.waitKey(0)
    return retVal


def testFun():
    tmpImgName = '../../img_test/test_sumammeter.png'
    # tmpImgName = '../../template_img/pointer_meter/template_sum_ammeter.png'
    destImgName = '../../img_test/test_sum_ammeter1.png'
    templateImg = cv2.imread(tmpImgName)
    destImg = cv2.imread(destImgName)
    tmpHeight = templateImg.shape[0]
    tmpWidth = templateImg.shape[1]
    destGray = cv2.cvtColor(destImg, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)

    # 模版检测--------------------------
    # maxval,t_left, b_right = get_match(templateImg, cv2.TM_CCOEFF_NORMED, destImg, tmpWidth, tmpHeight)

    # 边界检测--------------------------
    getMeterResult(templateGray, tmpWidth, tmpHeight, 10, 10)




def preProcessImg():
    inputImg = '../img_test/testc.png'
    outputImg = '../img_test_corrected/testc.png'
    tmpImg = '../template/tmpc.png'
    method = cv2.TM_CCOEFF_NORMED
    template = cv2.imread(tmpImg)
    gray = cv2.imread(inputImg, 0)
    # res = cv2.matchTemplate(gray, template, method)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

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


def detectRect(image, width, height):
    # 读取图片
    # image = cv2.imread('image.jpg')

    # 将图片转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义绿色的HSV范围
    lower_green = (60, 40, 40)
    upper_green = (70, 255, 255)

    # 根据HSV范围创建掩膜
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 在原始图像上应用掩膜
    result = cv2.bitwise_and(image, image, mask=mask)

    # 将结果转换为灰度图像
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # 使用边缘检测算法检测矩形轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓
    print('----------\n')
    i = 0
    foundAreas = []
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 绘制矩形边界框
        if w >= width and h >= height:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(str(i) + '----------(' + str(x) + ',' + str(y) + ')-------(' + str(x + w) + ',' + str(y + h) + ')----')
            # cv2.imshow('Result', image)
            # cv2.waitKey(0)
            foundAreas.append(contour)
            i += 1

    j = 0
    # 数组逆序，数组中就是按照顺序存放的识别结果
    realAreas = foundAreas[::-1]
    for area in realAreas:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(area)
        print(str(i) + '----------(' + str(x) + ',' + str(y) + ')-------(' + str(x + w) + ',' + str(y + h) + ')----')
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 识别区域的图片数据
        cv2.imshow('Result', image[y:y+h, x:x+w])
        cv2.waitKey(0)
        j += 1

    # 显示结果图像
    return realAreas
    # cv2.destroyAllWindows()




if __name__ == "__main__":
    testFun()
    # preProcessImg()
    # # 读取输入图片
    # mimage = cv2.imread("../../template_img/light_meter/template_light.png")
    # detectRect(mimage)
    # # 进行灯光检测
    # result = detect_light(image)
    print('------\n')
