import cv2
import numpy as np
from PIL import Image
# from template_process.utils.utils_qr import qr_decode, qr_encode
import template_process.utils.utils_qr as qr_utils
import template_process.pointer_meter.template_sub_ammeter as template_sub_ammeter
import template_process.light_meter.template_light as template_light

def get_match(template, method, img, width, height):

    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 设置匹配的阈值，可以根据实际情况调整
    threshold = 0.7
    # 使用np.where找到匹配结果大于阈值的位置
    locations = np.where(res >= threshold)
    # locations = list(zip(*locations[::-1]))
    # 循环遍历每个匹配结果的位置
    i = 0
    preX = -10
    preY = -10
    for (x, y) in zip(*locations[::-1]):
        # 在目标图片上绘制矩形框标记匹配区域
        if abs(x - preX) > 10:
            # print(str(i) + '----------(' + str(x) + ',' + str(y) + ')-------(' + str(x + width) + ',' + str(y + height) + ')----')
            cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 1)
            preX = x
            preY = y
            i += 1

    # 显示标记了匹配区域的目标图片
    # detectAreas = detectRect(img, width, height)
    # cv2.destroyAllWindows()

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    # print("----------------------------")
    # print("min_val", min_val)
    # print("max_val", max_val)
    # print("min_loc", min_loc)
    # print("max_loc", max_loc)
    # print("----------------------------")
    bottom_right = (top_left[0] + width, top_left[1] + height)
    return max_val, top_left, bottom_right


def GetComMeterValue(destImg, templateImg):
    print('---------GetComMeterValue-------------')
    tmpHeight = templateImg.shape[0]
    tmpWidth = templateImg.shape[1]
    destGray = cv2.cvtColor(destImg, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)

    originImg = destImg.copy()
    # 模版检测--------------------------
    maxval, t_left, b_right = get_match(templateImg, cv2.TM_CCOEFF_NORMED, destImg, tmpWidth, tmpHeight)
    # cv2.imshow('templateGray', destImg)
    # cv2.waitKey(0)
    # 监测指针表盘
    detectAreas = detectRect(destImg, tmpWidth, tmpHeight)
    pmeter_result = ''
    i = 0
    for area in detectAreas:
        x, y, w, h = cv2.boundingRect(area)
        # print(str(i) + '----------(' + str(x) + ',' + str(y) + ')-------(' + str(x + w) + ',' + str(y + h) + ')----')
        mVal = template_sub_ammeter.GetMeterValue(destImg[y:y + h, x:x + w], '')
        pmeter_result += str(mVal)
        if i < 2:
            pmeter_result += '_'
        i += 1
        # print('result = ' + str(result))

    # 监测灯状态
    light_result = template_light.get_2_light_result(originImg)
    print('----light_result----' + str(light_result))
    return pmeter_result, light_result


def testFun():
    tmpImgName = '../../img625/sum_ammeter_temp.png'
    destImgName = '../../img625/test_com_meter2.png'
    templateImg = cv2.imread(tmpImgName)
    destImg = cv2.imread(destImgName)
    tmpHeight = templateImg.shape[0]
    tmpWidth = templateImg.shape[1]
    destGray = cv2.cvtColor(destImg, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)

    originImg = destImg.copy()
    # 模版检测--------------------------
    cv2.imshow('templateGray', destImg)
    cv2.imshow('templateGray', templateImg)
    cv2.waitKey(0)
    maxval,t_left, b_right = get_match(templateImg, cv2.TM_CCOEFF_NORMED, destImg, tmpWidth, tmpHeight)
    cv2.imshow('templateGray', destImg)
    cv2.waitKey(0)
    # 监测指针表盘
    detectAreas = detectRect(destImg, tmpWidth, tmpHeight)
    i = 0
    for area in detectAreas:
        x, y, w, h = cv2.boundingRect(area)
        print(str(i) + '----------(' + str(x) + ',' + str(y) + ')-------(' + str(x + w) + ',' + str(y + h) + ')----')
        result = template_sub_ammeter.GetMeterValue(destImg[y:y+h, x:x+w], '')
        print('result = ' + str(result))

    # 监测灯状态
    light_result = template_light.get_2_light_result(originImg)
    print('----light_result----' + str(light_result))




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
#     # pp1 = (0,0)
#     # pp2 = (500, 500)
#     # cv2.line(template, pp1, pp2, (0,255, 0), cv2.LINE_AA)
#     # cv2.imshow('result111', template)
#     # cv2.waitKey(0)
#
#
#     # (startX, startY) = max_loc
#     # endX = startX + template.shape[1]
#     # endY = startY + template.shape[0]
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
    # 将图片转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 将BGR蓝色转换为HSV颜色空间
    blue_bgr = np.uint8([[[255, 0, 0]]])
    blue_hsv = cv2.cvtColor(blue_bgr, cv2.COLOR_BGR2HSV)
    # 定义蓝色范围
    lower_blue = np.array([blue_hsv[0][0][0] - 10, 100, 100])
    upper_blue = np.array([blue_hsv[0][0][0] + 10, 255, 255])
    # 根据HSV范围创建掩膜
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # 在原始图像上应用掩膜
    result = cv2.bitwise_and(image, image, mask=mask)
    # 将结果转换为灰度图像
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # 使用边缘检测算法检测矩形轮廓
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 遍历所有轮廓
    # print('----------\n')
    i = 0
    foundAreas = []
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 绘制矩形边界框
        if w >= width and h >= height:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # print(str(i) + '----------(' + str(x) + ',' + str(y) + ')-------(' + str(x + w) + ',' + str(y + h) + ')----')
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
        # print(str(i) + '----------(' + str(x) + ',' + str(y) + ')-------(' + str(x + w) + ',' + str(y + h) + ')----')
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 识别区域的图片数据
        # cv2.imshow('Result', image[y:y+h, x:x+w])
        # cv2.waitKey(0)
        j += 1
    # 显示结果图像
    return realAreas


if __name__ == "__main__":
    testFun()
    # # 读取输入图片
    # mimage = cv2.imread("../../template_img/light_meter/template_light.png")
    # detectRect(mimage)
    # # 进行灯光检测
    # result = detect_light(image)
    print('------\n')
