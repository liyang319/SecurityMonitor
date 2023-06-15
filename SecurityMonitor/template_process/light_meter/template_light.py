import cv2
import numpy as np
from PIL import Image


def detect_light(image):
    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 红色的HSV范围
    lower_red = np.array([0, 100, 180])
    upper_red = np.array([20, 255, 255])
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # 绿色的HSV范围
    lower_green = np.array([40, 100, 180])
    upper_green = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # 计算红色和绿色区域的像素数量
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    # 判断哪个灯点亮
    if red_pixels > green_pixels:
        return "红色灯点亮"
    elif green_pixels > red_pixels:
        return "绿色灯点亮"
    else:
        return "没有灯点亮"


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
    image = cv2.imread("../../template_img/light_meter/template_light_red.png")
    # 进行灯光检测
    result = detect_light(image)
    print('------\n')
