import numpy as np
import cv2
import os
import template_process.com_meter.com_meter1 as com_meter1_process
import template_process.com_meter.com_meter2 as com_meter2_process
import template_process.utils.deviceInfo_class as DeviceInfo
import json


def CorrectImage(queryImagePath, templateImgDir, outImg, val_num=100, threshold=90):
    """
    Find the template class for the query image and to correct the query image
    :param queryImagePath: the path of the query image  eg. "./img_test/test3.png"
    :param templateImgDir: the dir of the template      eg. "./template/"
    :param outImg:  the out put dir of corrected image  eg. "./img_test_corrected/"
    :param val_num: the number of samples for validating the Homography matrix eg. val_num=100
    :param threshold: the error threshold of the Homography mapping
                      from A to B using Homography_matrix. Suggest:
                      30<= threshold <=100 (Note that the results we got after the experiment
                      by Statistically matching template mean)
    :return the class of template or None
    """

    # queryImage
    queryImage = cv2.imread(queryImagePath, 0)

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors of queryImage with SIFT
    queryImageKP, queryImageDES = sift.detectAndCompute(queryImage, None)

    # template images
    result = []  # [{"template_class": 1, "template_filename": "class1.png",
    # "homography_matrix": narray() }]
    for templateImgName in os.listdir(templateImgDir):
        # get the keypoints and descriptors of templateImage with SIFT
        if not templateImgName.endswith('.png'):
            continue
        templateImgPath = templateImgDir + templateImgName
        templateImg = cv2.imread(templateImgPath, 0)
        print('------' + templateImgPath)
        templateImgKP, templateImgDES = sift.detectAndCompute(templateImg, None)

        # match the keypoints
        bfMatcher = cv2.BFMatcher(crossCheck=True)
        matches = bfMatcher.match(queryImageDES, templateImgDES)
        matchesSorted = sorted(matches, key=lambda x: x.distance)

        """
        choose the first four matches to compute the Homography matrix
        and other 100 keypoints to validate the Homography matrix.
        """
        matchesForHMatrix = matchesSorted[:4]
        matchesForValidateH = matchesSorted[4:4 + val_num]

        # get the Homography matrix
        src_points = []
        target_points = []
        for match in matchesForHMatrix:
            query_index = match.queryIdx
            src_points.append(queryImageKP[query_index].pt)
            template_index = match.trainIdx
            target_points.append(templateImgKP[template_index].pt)
        hMatrix, s = cv2.findHomography(np.float32(src_points), np.float32(target_points), cv2.RANSAC, 10)

        # statistical the val set to find matching points to compute
        # the ratio of suitability
        error_total = 0
        for valMatche in matchesForValidateH:
            valsrc_index = valMatche.queryIdx
            valsrc_point = queryImageKP[valsrc_index].pt
            valsrc_point = valsrc_point + (1,)
            valtarget_index = valMatche.trainIdx
            valtarget_point = templateImgKP[valtarget_index].pt
            valtarget_point = valtarget_point + (1,)
            valsrc_point = np.array(valsrc_point)
            valtarget_point = np.array(valtarget_point)

            # b = H * aT
            error = np.sum(np.abs(valtarget_point - np.matmul(hMatrix, valsrc_point)))
            error_total = error_total + error

        if error_total / val_num < threshold:  # maybe change the threshold
            # finded the right template
            template_finded = {"template_class": templateImgName.split(".")[0],
                               "template_filename": templateImgName,
                               "homography_matrix": hMatrix}
            result.append(template_finded)
        # Draw first 10 matches.
        # imgShow = cv2.drawMatches(queryImage, queryImageKP, templateImg,
        #                          templateImgKP, matchesSorted[:10], None, flags=2)
        # plt.imshow(imgShow), plt.show()
        # cv2.findHomography()

    if len(result) == 0:
        print("no find the correct template")
        return None
    if len(result) > 1:
        print("warring: there are two templates that match the query image and we just return one")

    # template class
    result_tamplate_class = result[0]["template_class"]

    # correct the query img
    corrected_img = cv2.warpPerspective(queryImage, result[0]["homography_matrix"], queryImage.shape)
    cv2.imwrite(outImg + queryImagePath.split("/")[-1], corrected_img)

    return result_tamplate_class


def detectRect(inputImage, width, height):
    image = inputImage.copy()
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


def get_match(template, method, img, width, height):

    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # 设置匹配的阈值，可以根据实际情况调整
    threshold = 0.5
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
            cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
            i += 1
            preX = x
            preY = y

    # 显示标记了匹配区域的目标图片
    # detectAreas = detectRect(img, width, height)

    # cv2.imshow('Matched Objects', img)
    # cv2.waitKey(0)
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


def testFun():
    tmpImgName = './img625/com_meter1.png'
    destImgName = './img625/multi_com_meter.png'
    templateImg = cv2.imread(tmpImgName)
    destImg = cv2.imread(destImgName)
    tmpHeight = templateImg.shape[0]
    tmpWidth = templateImg.shape[1]
    destGray = cv2.cvtColor(destImg, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
    originImg = destImg.copy()
    tmpSubMeterName = './img_new/sub_ammeter2.png'
    tmpSubMeterImg = cv2.imread(tmpSubMeterName)

    maxval,t_left, b_right = get_match(templateImg, cv2.TM_CCOEFF_NORMED, destImg, tmpWidth, tmpHeight)

    detectAreas = detectRect(destImg, tmpWidth, tmpHeight)

    i = 0
    for area in detectAreas:
        print('----------COMMETER------------')
        x, y, w, h = cv2.boundingRect(area)
        # print(str(i) + '----------(' + str(x) + ',' + str(y) + ')-------(' + str(x + w) + ',' + str(y + h) + ')----')
        com_meter1_process.GetComMeterValue(originImg[y:y + h, x:x + w], tmpSubMeterImg)
        # print('result = ' + str(result))
        # cv2.imshow('Matched Objects', originImg[y:y + h, x:x + w])
        # cv2.waitKey(0)


def ProcessComMeter1():
    # commeter1先按模板匹配，取第一个
    retVal = 'NONE'
    tmpImgName = './img625/com_meter1.png'
    destImgName = './img625/multi_com_meter.png'
    templateImg = cv2.imread(tmpImgName)
    destImg = cv2.imread(destImgName)
    tmpHeight = templateImg.shape[0]
    tmpWidth = templateImg.shape[1]
    destGray = cv2.cvtColor(destImg, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
    originImg = destImg.copy()
    tmpSubMeterName = './img_new/sub_ammeter2.png'
    tmpSubMeterImg = cv2.imread(tmpSubMeterName)

    maxval, t_left, b_right = get_match(templateImg, cv2.TM_CCOEFF_NORMED, destImg, tmpWidth, tmpHeight)

    detectAreas = detectRect(destImg, tmpWidth, tmpHeight)

    i = 0
    for area in detectAreas:
        # print('----------COMMETER------------')
        x, y, w, h = cv2.boundingRect(area)
        pval, lval = com_meter1_process.GetComMeterValue(originImg[y:y + h, x:x + w], tmpSubMeterImg)
        # print('----pval=' + str(pval) + '---lval=' + str(lval))
        return pval, lval
        # 每次检测只关注一组仪表，所以不在识别多余的匹配

    return 'NONE', 'NONE'


def ProcessComMeter2():
    # commeter2没有做模板匹配
    retVal = ''
    tmpImgName = './img625/com_meter1.png'
    destImgName = './img625/test_com_meter2.png'
    templateImg = cv2.imread(tmpImgName)
    destImg = cv2.imread(destImgName)
    tmpHeight = templateImg.shape[0]
    tmpWidth = templateImg.shape[1]
    destGray = cv2.cvtColor(destImg, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(templateImg, cv2.COLOR_BGR2GRAY)
    originImg = destImg.copy()
    tmpSubMeterName = './img625/sum_ammeter_temp.png'
    tmpSubMeterImg = cv2.imread(tmpSubMeterName)

    pval, lval = com_meter2_process.GetComMeterValue(originImg, tmpSubMeterImg)
    return pval, lval

def DoRecognization(devInfo):
    retVal = ''
    if devInfo.meterType == 'ComMeter2':
        pval, lval = ProcessComMeter1()
        devInfo.SetPointerMeterVal(pval)
        devInfo.SetLightMeterVal(lval)
    else:
        pval, lval = ProcessComMeter2()
        devInfo.SetPointerMeterVal(pval)
        devInfo.SetLightMeterVal(lval)
    return retVal



if __name__ == "__main__":
    queryImagePath = "./img_test/test1.png"  # the image to be corrected
    templateImgDir = "./template/"  # the tamplate dir
    outImg = "./img_test_corrected/"

    # testFun()
    ProcessComMeter2()
    # find the corresponding template and correct the img
    # matchedTemplateClass = CorrectImage(queryImagePath, templateImgDir, outImg)
