import cv2
import numpy as np
from PIL import Image
from enum import Enum


class LIGHT_TYPE(Enum):
    LIGHT_NONE = 'LIGHT_NONE'
    LIGHT_RED = 'LIGHT_RED'
    LIGHT_GREEN = 'LIGHT_GREEN'
    LIGHT_YELLOW = 'LIGHT_YELLOW'


# def get_2_light_result(image):
#     # 将图像转换为HSV颜色空间
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     print(str(hsv_image.size))
#     # cv2.imshow('Matched Objects', hsv_image)
#     # cv2.waitKey(0)
#     # 红色的HSV范围         红色灭的色值区间
#     lower_red = np.array([140, 180, 120])
#     upper_red = np.array([190, 220, 150])
#     red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
#
#     # 绿色的HSV范围         绿色灭的色值区间
#     lower_green = np.array([50, 210, 70])
#     upper_green = np.array([90, 255, 110])
#     green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
#
#     # 计算红色和绿色区域的像素数量
#     red_pixels = cv2.countNonZero(red_mask)
#     green_pixels = cv2.countNonZero(green_mask)
#     print('red=' + str(red_pixels) + ' green=' + str(green_pixels))
#     # 判断哪个灯点亮
#     if red_pixels > green_pixels:
#         return LIGHT_TYPE.LIGHT_GREEN
#     elif green_pixels > red_pixels:
#         return LIGHT_TYPE.LIGHT_RED
#     else:
#         return LIGHT_TYPE.LIGHT_NONE

def get_2_light_result(image):
    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print(str(hsv_image.size))
    # cv2.imshow('Matched Objects', hsv_image)
    # cv2.waitKey(0)

    # 定义红色的HSV范围
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv_image, lower_red, upper_red)

    # 合并两个红色区域的掩码
    red_mask = red_mask1 + red_mask2

    red_pixels = cv2.countNonZero(red_mask)
    print('red_pixels=' + str(red_pixels))

    # 定义绿色的HSV范围
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    green_pixels = cv2.countNonZero(green_mask)
    print('green_pixels=' + str(green_pixels))

    # 计算红色和绿色区域的像素数量
    # red_pixels = cv2.countNonZero(red_mask)
    # green_pixels = cv2.countNonZero(green_mask)
    print('red=' + str(red_pixels) + ' green=' + str(green_pixels))
    # 判断哪个灯点亮
    if red_pixels > green_pixels:
        return LIGHT_TYPE.LIGHT_RED
    elif green_pixels > red_pixels:
        return LIGHT_TYPE.LIGHT_GREEN
    else:
        return LIGHT_TYPE.LIGHT_NONE


def test_light2(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # red
    lower_hsv_red = np.array([157, 177, 122])  # 红上
    upper_hsv_red = np.array([179, 255, 255])  # 红下
    mask_red = cv2.inRange(hsv, lowerb=lower_hsv_red, upperb=upper_hsv_red)  # 过滤出红色部分
    # 中值滤波
    red_blur = cv2.medianBlur(mask_red, 7)
    # green
    lower_hsv_green = np.array([49, 79, 137])
    upper_hsv_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lowerb=lower_hsv_green, upperb=upper_hsv_green)
    # 中值滤波
    green_blur = cv2.medianBlur(mask_green, 7)

    # 因为图像是二值的图像，所以如果图像出现白点，也就是255，那么就取他的max最大值255
    red_color = np.max(red_blur)
    green_color = np.max(green_blur)
    # 在red_color中判断二值图像如果数值等于255，那么就判定为red
    if red_color == 255:
        print('red')
        cv2.rectangle(image, (660, 420), (400, 300), (0, 0, 255), 2)  # 按坐标画出矩形框
        cv2.putText(image, "red", (500, 280), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # 显示red文本信息
    # 在green_color中判断二值图像如果数值等于255，那么就判定为green
    elif green_color == 255:
        print('green')
        cv2.rectangle(image, (660, 420), (400, 300), (0, 0, 255), 2)
        cv2.putText(image, "green", (500, 280), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # cv2.imshow('frame', hsv)  # 输出
    # c = cv2.waitKey(0)


def checkRed(image):
    # 加载图像
    # image = cv2.imread('../../template_img/light_meter/template_light2.png')

    # 转换图像颜色空间为HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义红色的HSV范围
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red, upper_red)

    # 合并两个红色区域的掩码
    mask = mask1 + mask2

    red_pixels = cv2.countNonZero(mask)
    print('red_pixels=' + str(red_pixels))
    # 对原始图像应用掩码
    red_image = cv2.bitwise_and(image, image, mask=mask)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Red Image', red_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def checkGreen(image):
    # 加载图像

    # 转换图像颜色空间为HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义绿色的HSV范围
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    green_pixels = cv2.countNonZero(mask)
    print('green_pixels=' + str(green_pixels))
    # 对原始图像应用掩码
    green_image = cv2.bitwise_and(image, image, mask=mask)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Green Image', green_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def checkOnOff(image):
    # 提取灯的区域
    # 这里需要根据实际情况调整灯的位置和大小
    # 可以使用图像处理工具（如Photoshop）查看目标图片并获取灯的位置信息
    height, width, _ = image.shape
    # light_roi = image[0:150, 200:250]
    light_roi = image

    # 将灯的区域转换为灰度图像
    gray_roi = cv2.cvtColor(light_roi, cv2.COLOR_BGR2GRAY)

    # 对灰度图像进行阈值处理
    # 这里需要根据实际情况调整阈值的值
    # 如果灯亮时灰度值较高，可以使用较高的阈值
    # 如果灯亮时灰度值较低，可以使用较低的阈值
    _, thresholded_roi = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY)

    # 统计灯亮的像素数量
    light_pixel_count = cv2.countNonZero(thresholded_roi)

    # 判断灯是否点亮
    if light_pixel_count > 0:
        print('灯亮')
    else:
        print('灯灭')


def test_light():
    # 读取图片
    image = Image.open('../../template_img/light_meter/template_light2.png')
    # 定义红绿灯颜色的阈值范围
    red_threshold = (150, 0, 0, 255)
    green_threshold = (0, 150, 0, 255)

    # 获取图片的像素数据
    pixels = image.load()

    # 获取图片的尺寸
    width, height = image.size

    # 判断红绿灯的状态
    red_light = False
    green_light = False

    # 遍历图片的所有像素
    for x in range(width):
        for y in range(height):
            pixel = pixels[x, y]
            if pixel < red_threshold:
                red_light = True
            elif pixel < green_threshold:
                green_light = True

    # 打印红绿灯的状态
    if red_light:
        print("红灯亮")
    else:
        print("红灯灭")

    if green_light:
        print("绿灯亮")
    else:
        print("绿灯灭")



def get_match(template, method, img, width, height):

    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 设置匹配的阈值，可以根据实际情况调整
    threshold = 0.55
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
    # detectAreas = detectRect(img, width, height)

    cv2.imshow('Matched Objects', img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

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


def testFun():
    tmpImgName = '../../img_new/green_off.png'
    destImgName = '../../img_new/single_com_meter.png'
    templateImg = cv2.imread(tmpImgName)
    destImg = cv2.imread(destImgName)
    tmpHeight = templateImg.shape[0]
    tmpWidth = templateImg.shape[1]
    destGray = cv2.cvtColor(destImg, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
    # ret, thresh1 = cv2.threshold(destGray, 70, 120, cv2.THRESH_BINARY)
    # circles = cv2.HoughCircles(destGray, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=200, maxRadius=0)
    # edges = cv2.Canny(destGray, 110, 255, apertureSize=3)
    # 模版检测--------------------------
    maxval,t_left, b_right = get_match(templateImg, cv2.TM_CCOEFF_NORMED, destImg, tmpWidth, tmpHeight)
    # cv2.imshow('templateGray', edges)
    # cv2.waitKey(0)




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

    # 读取输入图片
    # image = cv2.imread("../../img_new/green.png")
    image = cv2.imread("../../img_new/test_light2.png")
    # cv2.imshow('templateGray', image)
    # cv2.waitKey(0)
    # 进行灯光检测
    result = get_2_light_result(image)
    # get_2_light_result(image)
    #  print('------\n' + str(result))
    # checkOnOff(image)
    # checkGreen(image)
    # checkRed(image)
