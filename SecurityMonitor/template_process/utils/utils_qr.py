import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode


def qr_decode():
    # 读取图片
    image = cv2.imread('../../img_test/qr_test.png')
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 解码二维码
    data = decode(gray)
    # 遍历解码结果
    for obj in data:
        # 提取二维码的边界框坐标
        x, y, w, h = obj.rect

        # 在图像上绘制边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 提取二维码数据
        qr_data = obj.data.decode('utf-8')

        # 在图像上显示二维码数据
        cv2.putText(image, qr_data, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 打印二维码数据
        print("二维码数据:", qr_data)

    # 显示图像
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    qr_decode()
    print('------\n')
