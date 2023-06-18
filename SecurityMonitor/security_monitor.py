import argparse
import pointer_meter_match
import importlib
from enum import Enum
import cv2
import template_process.utils.utils_qr as qr_utils
import json

meterTypeConfigFileName = './config/meter_type.json'
monitorConfigFileName = './config/monitor_config.json'
inputFileName = './img_test/qr_test.png'
g_QrCodeArray = []
finalResult = ''
g_MeterTypeObj = []
g_DeviceConfigObj = []

g_ExpResult = ''
g_ActResult = ''
g_fullResult = ''

class MonitorType(Enum):
    POINTER_METER = 'pointer_meter'
    LIGHT_METER = 'light_meter'


def find_templateclass_using_name(class_type, class_name):
    typeHeader = 'template_process.'
    if class_type == MonitorType.POINTER_METER:
        typeHeader += str(MonitorType.POINTER_METER.value)
    elif class_type == MonitorType.LIGHT_METER:
        typeHeader += str(MonitorType.LIGHT_METER.value)
    templateclass_name = typeHeader + '.' + str(class_name)
    print(templateclass_name)
    templateclass = importlib.import_module(templateclass_name)

    if templateclass is None:
        raise NotImplementedError("In templater_process package, the model %s not find." % (templateclass_name))

    return templateclass



def loadConfiguration():
    print('----loadConfiguration---')
    # 仪表类型配置
    with open(meterTypeConfigFileName, 'r') as file:
        # 读取文件内容
        dev_json_str = file.read()
        # 解析JSON字符串 声明全局变量
        global g_MeterTypeObj
        g_MeterTypeObj = json.loads(dev_json_str)
        print(g_MeterTypeObj['PointerMeter_1']['deviceType'])
    # 设备预警配置
    with open(monitorConfigFileName, 'r') as file:
        # 读取文件内容
        monitor_json_str = file.read()
        # 解析JSON字符串
        global g_DeviceConfigObj
        g_DeviceConfigObj = json.loads(monitor_json_str)

def detectQrCode(image):
    print('----detectQrCode---')
    qrArr = qr_utils.qr_decode(image)
    return qrArr


def getExpResult():
    retVal = g_DeviceConfigObj[g_QrCodeArray[0].val]['result']
    return retVal


def getActResult():
    print('----detectQrCode---')
    retVal = ''
    return retVal


def formatResult():
    retVal = ''
    return retVal


if __name__ == "__main__":
    loadConfiguration()
    image = cv2.imread(inputFileName)
    g_QrCodeArray = detectQrCode(image)
    print(g_DeviceConfigObj[g_QrCodeArray[0].val]['result'])
    # g_ExpResult = g_DeviceConfigObj[g_QrCodeArray[0].val]['result']
    g_ExpResult = getExpResult()
    g_ActResult = getActResult()
    g_fullResult = formatResult()
    print(g_fullResult)
    print('------\n')



    # queryImagePath = "./img_new/img00.png"  # the image to be corrected
    # templateImgDir = "./template_img/pointer_meter/"  # the tamplate dir
    # outImg = "./img_test_corrected/"
    # matchedTemplateClass = pointer_meter_match.CorrectImage(queryImagePath, templateImgDir, outImg)
    # templateclass = find_templateclass_using_name(MonitorType.POINTER_METER, matchedTemplateClass)
    # templateclass.testFun()


    # matchedTemplateClass = img_match.CorrectImage(queryImagePath, templateImgDir, outImg)
    # print(matchedTemplateClass)
    #
    # # check the pointer position and compute the num according to the degree of pointer
    # if matchedTemplateClass is None:
    #     raise ValueError("no find the right template class")
    #
    # corrected_img_path = outImg + queryImagePath.split("/")[-1]
    # templateclass = find_templateclass_using_name(matchedTemplateClass)
    #
    # num = templateclass.degree2num(corrected_img_path)
    # print(num)







# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--img_path", type=str, default="./img_test/test1.png",
#                     help='the path of the test image')
# parser.add_argument("--template_dir", type=str, default="./template/",
#                     help='the dir of template images')
# parser.add_argument("--siftedimg_dir", type=str, default="./img_test_corrected/",
#                     help='the dir of sifted images')
#
# opt, _ = parser.parse_known_args()
#
# # find the right template and correct the image
# queryImagePath = opt.img_path
# templateImgDir = opt.template_dir
# outImg = opt.siftedimg_dir
# print(opt)
# matchedTemplateClass = img_match.CorrectImage(queryImagePath, templateImgDir, outImg)
# print(matchedTemplateClass)
#
# # check the pointer position and compute the num according to the degree of pointer
# if matchedTemplateClass is None:
#     raise ValueError("no find the right template class")
#
# corrected_img_path = outImg + queryImagePath.split("/")[-1]
# templateclass = find_templateclass_using_name(matchedTemplateClass)
#
# num = templateclass.degree2num(corrected_img_path)
# print(num)
