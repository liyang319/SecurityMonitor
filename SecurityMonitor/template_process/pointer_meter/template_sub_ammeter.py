import cv2
import numpy as np
from PIL import Image
import math

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


def enhance_contrast(image, alpha, beta):
    # 线性变换的公式为：output = alpha * input + beta
    # 将图像转换为浮点数类型
    image = image.astype(float)
    # 线性变换
    enhanced_image = alpha * image + beta
    # 限制像素值范围在0到255之间
    enhanced_image = np.clip(enhanced_image, 0, 255)
    # 转换为无符号整数类型
    enhanced_image = enhanced_image.astype(np.uint8)
    return enhanced_image


def getMeterResult(dstImg, width, height, wOffset, hOffset):
    retVal = 0
    enhanced_image = enhance_contrast(dstImg, 0.8, 0)
    # cv2.imshow('enhance', enhanced_image)
    # cv2.waitKey(0)

    ret, thresh1 = cv2.threshold(enhanced_image, 130, 255, cv2.THRESH_BINARY)
    # 二值化后 分割主要区域 减小干扰 模板图尺寸656*655
    tm = thresh1.copy()
    test_main = tm[hOffset:(height - hOffset), wOffset:(width - wOffset)]

    # 边缘化检测
    edges = cv2.Canny(test_main, 50, 255, apertureSize=3)

    # 霍夫直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
    if lines is None:
        # print('No Line')
        retVal = 0
    result = edges.copy()
    # cv2.imshow('result111', result)
    # cv2.waitKey(0)

    if lines is not None and lines.size != 0:
        for line in lines[0]:
            rho = line[0]  # 第一个元素是距离rho
            theta = line[1]  # 第二个元素是角度theta
            # print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
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
            # 计算表针读数
            angle = get_angle(pt1[0], pt1[1], pt2[0], pt2[1])
            # print('angle === ' + str(angle))
            retVal = caculateMeterVal(angle, 0, 600)
            # print(str(retVal))
    else:
        retVal = 0
        # print('There is no pointer detected!!!')
    # cv2.imwrite('../../img_test_corrected/test_sum11.png', result)
    # cv2.imshow('result111', result)
    # cv2.waitKey(0)

    return round(retVal, 2)


def get_angle(x1, y1, x2, y2):
    if x1 == x2:
        if y2 > y1:
            return 90
        else:
            return 270
    else:
        slope = (y2 - y1) / (x2 - x1)
        angle = math.atan(slope)
        return math.degrees(angle)


def caculateMeterVal(angle, minVal, maxVal):
    retVal = 0
    retVal = (angle / 90) * (maxVal - minVal)
    return retVal


def GetMeterValue(dstImg, params):
    # templateImg = cv2.imread(dstImgName)
    dstImgGray = cv2.cvtColor(dstImg, cv2.COLOR_BGR2GRAY)
    dstImgHeight = dstImg.shape[0]
    dstImgWidth = dstImg.shape[1]
    retVal = getMeterResult(dstImgGray, dstImgWidth, dstImgHeight, 40, 40)
    # print('GetMeterValue = ' + str(retVal))
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




# def preProcessImg():
#     inputImg = '../img_test/testc.png'
#     outputImg = '../img_test_corrected/testc.png'
#     tmpImg = '../template/tmpc.png'
#     method = cv2.TM_CCOEFF_NORMED
#     template = cv2.imread(tmpImg)
#     gray = cv2.imread(inputImg, 0)
#     # res = cv2.matchTemplate(gray, template, method)
#     # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#     # 边缘化检测
#     edges = cv2.Canny(template, 50, 150, apertureSize=3)
#     # 霍夫直线
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 60)
#     if lines is None:
#         print("---")
#     result = edges.copy()
#
#     # cv2.rectangle(gray, (startX, startY), (endX, endY), (255, 0, 0), 3)
#     for line in lines[0]:
#         rho = line[0]  # 第一个元素是距离rho
#         theta = line[1]  # 第二个元素是角度theta
#         print('distance:' + str(rho), 'theta:' + str(((theta / np.pi) * 180)))
#         lbael_text = 'distance:' + str(round(rho)) + 'theta:' + str(round((theta / np.pi) * 180 - 90, 2))
#         # cv2.putText(template, lbael_text, (t_left[0], t_left[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         if (theta > 3 * (np.pi / 3)) or (theta < (np.pi / 2)):  # 从图像边界画出延长直线
#             # 该直线与第一行的交点
#             pt1 = (int(rho / np.cos(theta)), 0)
#             # 该直线与最后一行的焦点
#             pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
#             # 绘制一条白线
#             # cv2.line(template, pt1, pt2, (0, 255, 0), cv2.LINE_AA)
#             # print('theat >180 theta<90')
#             cv2.circle(template, pt1, 40, (255, 255, 0), -1)  # 画圆形
#             cv2.circle(template, pt2, 40, (255, 255, 0), -1)  # 画圆形
#         else:  # 水平直线
#             # 该直线与第一列的交点
#             pt1 = (0, int(rho / np.sin(theta)))
#             # 该直线与最后一列的交点
#             pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
#             # 绘制一条直线
#             cv2.line(result, pt1, pt2, 255, 1)
#
#     cv2.imshow('result111', template)
#     cv2.waitKey(0)
#     print('------')



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
    dstImg = cv2.imread('../../img_new/sub_ammeter3.png')
    GetMeterValue(dstImg, '')
    # corrected_img_path = "../../img_test_corrected/test_sum11.png"
    # degree = degree2num(corrected_img_path)
    # preProcessImg()
    # # 读取输入图片
    # mimage = cv2.imread("../../template_img/light_meter/template_light.png")
    # detectRect(mimage)
    # # 进行灯光检测
    # result = detect_light(image)
    print('------\n')
